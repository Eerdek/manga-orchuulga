# app.py
from __future__ import annotations
import argparse, io, os, json, base64, zipfile, pathlib, traceback
from typing import List, Dict, Any, Tuple
from flask import Flask, request, Response, send_file, send_from_directory
from PIL import Image
import numpy as np

# ----- project deps -----
from utils import OCRBox, erase_text_area, inpaint_with_mask
from renderer import render_translations_styled, render_translations
from ocr_torii import ocr_image
from translate_lingva import translate_lines, translate_lines_impl
from config import DEFAULT_FONT_PATH

app = Flask(__name__, static_url_path="/static")
IN_DIR  = pathlib.Path("input");  IN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = pathlib.Path("output"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- watermark (optional) ----------
WATERMARK_PATH = os.environ.get("WATERMARK_PATH", r"C:\FEDoUP\watermark.png")
WM_RATIO   = float(os.environ.get("WM_RATIO", "0.22"))
WM_MIN_W   = int(os.environ.get("WM_MIN_W", "100"))
WM_OFFSET  = os.environ.get("WM_OFFSET", "+12+12")  # "+X+Y" bottom-right
WM_OPACITY = float(os.environ.get("WM_OPACITY", "1.0"))

def _parse_offset(s: str) -> Tuple[int,int]:
    s = s.strip()
    if s.startswith("+"): s = s[1:]
    try:
        dx, dy = [int(v) for v in s.split("+", 1)]
    except Exception:
        dx, dy = 12, 12
    return dx, dy

def apply_watermark(img: Image.Image,
                    wm_path=WATERMARK_PATH,
                    ratio=WM_RATIO,
                    min_w=WM_MIN_W,
                    offset=WM_OFFSET,
                    opacity=WM_OPACITY) -> Image.Image:
    if not wm_path or not os.path.exists(wm_path):
        return img
    base = img.convert("RGBA")
    W, H = base.size
    wm = Image.open(wm_path).convert("RGBA")
    target_w = max(min_w, int(W * ratio))
    scale = target_w / max(1, wm.width)
    wm = wm.resize((target_w, max(1, int(wm.height * scale))), Image.LANCZOS)
    if opacity < 1.0:
        a = wm.split()[-1].point(lambda p: int(p * opacity))
        wm.putalpha(a)
    dx, dy = _parse_offset(offset)
    x = max(0, W - wm.width - dx)
    y = max(0, H - wm.height - dy)
    out = Image.new("RGBA", (W, H))
    out.paste(base, (0, 0))
    out.alpha_composite(wm, (x, y))
    return out

# ---------- helpers ----------
def polygon_to_bbox(poly):
    xs = poly[0::2]; ys = poly[1::2]
    x1, x2 = min(xs), max(xs); y1, y2 = min(ys), max(ys)
    return (int(x1), int(y1), int(x2-x1), int(y2-y1))

def _paths(layout_path: str, src_img: str):
    out_dir = pathlib.Path(layout_path).parent
    stem = pathlib.Path(src_img).stem
    clean = out_dir / f"{stem}.clean.png"
    edited = out_dir / f"{stem}.edited.png"
    mask_dir = out_dir / f"{stem}.masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    return pathlib.Path(src_img), clean, edited, mask_dir

def load_layout(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        j = json.load(f)
    j["boxes"] = [OCRBox(text=b["text"], bbox=tuple(b["bbox"])) for b in j.get("boxes", [])]
    j["translations"] = j.get("translations", [""] * len(j["boxes"]))
    j["styles"] = j.get("styles", [{"fontSize": None, "color": None} for _ in j["boxes"]])
    j["erasers"] = j.get("erasers", [])
    return j

def save_layout(p: str, d: Dict[str, Any]) -> None:
    out = {
        "source_image": d["source_image"],
        "boxes": [{"text": b.text, "bbox": list(b.bbox)} for b in d["boxes"]],
        "translations": d.get("translations", []),
        "styles": d.get("styles", []),
        "erasers": d.get("erasers", []),
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def _ensure_clean(layout_path: str, data: Dict[str, Any], force=False) -> str:
    src, clean, _, _ = _paths(layout_path, data["source_image"])
    if clean.exists() and not force:
        return str(clean)

    base = Image.open(src).convert("RGBA")

    # Auto erase OCR boxes unless manual erasers exist
    auto_erase = os.getenv("AUTO_ERASE", "1") not in ("0", "false", "False", "no", "No")
    erasers = list(data.get("erasers", []))

    if auto_erase and not erasers:
        for b in data.get("boxes", []):
            try:
                # Stronger cleaning: text-aware mask + aggressive=3
                erase_text_area(
                    base,
                    box=tuple(map(int, b.bbox)),
                    expand=18,
                    feather=3,
                    corner=8,
                    aggressive=3,
                    mask_mode="text",
                    pattern=None
                )
            except Exception:
                pass
    else:
        for e in erasers:
            if e.get("type") == "bbox":
                bbox = tuple(e.get("bbox", [0, 0, 0, 0]))
                params = e.get("params") or {}
                erase_text_area(
                    base, box=bbox,
                    expand=int(params.get("expand", 14)),
                    feather=int(params.get("feather", 3)),
                    corner=int(params.get("corner", 6)),
                    aggressive=int(params.get("aggressive", 2)),
                    mask_mode=e.get("mode", "auto"),
                    pattern=params.get("pattern")
                )
            elif e.get("type") == "mask":
                mf = e.get("mask_file")
                if not mf or not pathlib.Path(mf).exists():
                    continue
                mask = Image.open(mf).convert("L")
                mu8 = np.array(mask, dtype=np.uint8)
                ys, xs = np.where(mu8 > 0)
                if xs.size and ys.size:
                    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
                    out = inpaint_with_mask(base, mu8, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
                    if out is not None:
                        base.paste(out)

    base.save(clean, "PNG")
    return str(clean)

def _extract_boxes(ocr_json: Dict[str, Any]) -> List[OCRBox]:
    boxes: List[OCRBox] = []
    if not ocr_json: return boxes
    if "lines" in ocr_json:
        for l in ocr_json["lines"]:
            txt = (l.get("text") or "").strip()
            bb = l.get("boundingBox")
            if not txt or not bb or len(bb) != 8: continue
            x,y,w,h = polygon_to_bbox(bb)
            boxes.append(OCRBox(text=txt, bbox=(x,y,w,h)))
        if boxes: return boxes
    if "data" in ocr_json:
        for it in ocr_json["data"]:
            txt = (it.get("text") or "").strip()
            poly = it.get("polygon")
            if not txt or not poly or len(poly) < 8: continue
            x,y,w,h = polygon_to_bbox(poly)
            boxes.append(OCRBox(text=txt, bbox=(x,y,w,h)))
    return boxes

# ---------- pages ----------
@app.get("/")
def home():
    return Response(HOME_HTML, mimetype="text/html")

@app.get("/editor")
def editor():
    return Response(EDITOR_HTML, mimetype="text/html")

# ---------- static/img ----------
@app.get("/api/img")
def api_img():
    p = request.args.get("path","").strip()
    if not p: return {"error":"missing path"}, 400
    ap = pathlib.Path(p).resolve()
    if not ap.exists(): return {"error":"not found"}, 404
    return send_from_directory(ap.parent, ap.name)

@app.get("/api/exists")
def api_exists():
    p = request.args.get("path","").strip()
    return {"exists": pathlib.Path(p).exists()} if p else {"exists": False}

# ---------- upload ----------
@app.post("/api/upload")
def api_upload():
    fs = request.files.getlist("files")
    if not fs: return {"error":"no files"}, 400
    saved=[]
    for f in fs:
        name = pathlib.Path(f.filename).name
        if not name: continue
        path = IN_DIR / name
        f.save(path)
        saved.append(path.resolve().as_posix())
    return {"ok":True, "files": saved}

# ---------- auto translate ----------
@app.post("/api/auto_translate")
def api_auto_translate():
    j = request.get_json(force=True, silent=True) or {}
    images: List[str] = j.get("images") or []
    manga_mode = bool(j.get("manga_mode", True))
    max_chars = int(j.get("max_chars", 800))
    api_key = os.getenv("TORII_API_KEY")

    results=[]
    for ip in images:
        try:
            src = pathlib.Path(ip)
            if not src.exists():
                results.append({"image": ip, "error":"not found"}); continue
            with Image.open(src) as im: _ = im.size

            ocr = ocr_image(str(src), api_key)
            boxes = _extract_boxes(ocr)

            src_lines = [b.text.strip()[:max_chars] for b in boxes]
            if manga_mode:
                trans = translate_lines_impl(src_lines, reflow=False)
            else:
                try: trans = translate_lines(src_lines, reflow=False)
                except Exception: trans = translate_lines_impl(src_lines, reflow=False)

            while len(trans) < len(boxes):
               trans.append("")
            trans = trans[:len(boxes)]

            out_json = OUT_DIR / f"{src.stem}.layout.json"
            layout = {
                "source_image": str(src),
                "boxes": [{"text": b.text, "bbox": list(b.bbox)} for b in boxes],
                "translations": trans,
                "styles": [{"fontSize": None, "color": None} for _ in boxes],
                "erasers": [],
            }
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(layout, f, ensure_ascii=False, indent=2)

            clean_path = _ensure_clean(str(out_json), load_layout(str(out_json)), force=True)
            font_path = DEFAULT_FONT_PATH
            try:
                img = render_translations_styled(clean_path, boxes, trans, layout["styles"], font_path)
            except Exception:
                img = render_translations(str(src), boxes, trans, font_path)
            edited_path = OUT_DIR / f"{src.stem}.edited.png"
            img.save(edited_path, "PNG", optimize=False)

            results.append({
                "image": src.resolve().as_posix(),
                "layout": out_json.resolve().as_posix(),
                "clean": pathlib.Path(clean_path).resolve().as_posix(),
                "edited": edited_path.resolve().as_posix(),
            })
        except Exception as ex:
            results.append({"image": ip, "error": str(ex), "trace": traceback.format_exc()})
    return {"ok":True, "results": results}

# ---------- open layout ----------
@app.get("/api/open")
def api_open():
    lp = request.args.get("layout","").strip()
    if not lp or not pathlib.Path(lp).exists():
        return {"error":"layout not found"}, 400
    d = load_layout(lp)
    while len(d["styles"]) < len(d["boxes"]):
        d["styles"].append({"fontSize": None, "color": None})
    _, clean, edited, _ = _paths(lp, d["source_image"])    
    return {
        "source_image": d["source_image"],
        "boxes": [{"text": b.text, "bbox": list(b.bbox)} for b in d["boxes"]],
        "translations": d.get("translations", []),
        "styles": d.get("styles", []),
        "erasers": d.get("erasers", []),
        "clean_image": str(clean),
        "edited_image": str(edited),                        
    }

# ---------- brush(mask) erase ----------
@app.post("/api/erase_mask")
def api_erase_mask():
    j = request.get_json(force=True, silent=True) or {}
    lp = (j.get("layout_path") or "").strip()
    data_url = j.get("mask_data_url") or ""
    if not lp or not pathlib.Path(lp).exists():
        return {"error":"layout not found"}, 400
    if not data_url.startswith("data:image/png;base64,"):
        return {"error":"invalid mask"}, 400

    d = load_layout(lp)
    src, _, _, mask_dir = _paths(lp, d["source_image"])
    b64 = data_url.split(",",1)[1]
    mask_im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
    W0, H0 = Image.open(src).size
    if mask_im.size != (W0, H0):
        mask_im = mask_im.resize((W0, H0), Image.NEAREST)
    idx = sum(1 for e in d.get("erasers",[]) if e.get("type")=="mask") + 1
    mpath = mask_dir / f"mask_{idx:03d}.png"
    mask_im.save(mpath, "PNG")
    d["erasers"].append({"type":"mask", "mask_file": str(mpath)})
    save_layout(lp, d)
    clean_path = _ensure_clean(lp, d, force=True)
    return {"ok":True, "clean_image": clean_path, "mask_file": str(mpath)}

# ---------- rebuild clean ----------
@app.post("/api/rebuild_clean")
def api_rebuild_clean():
    j = request.get_json(force=True, silent=True) or {}
    lp = (j.get("layout_path") or "").strip()
    if not lp or not pathlib.Path(lp).exists():
        return {"error":"layout not found"}, 400
    d = load_layout(lp)
    clean_path = _ensure_clean(lp, d, force=True)
    return {"ok":True, "clean_image": clean_path}

# ---------- style / render ----------
@app.post("/api/update_style")
def api_update_style():
    j = request.get_json(force=True, silent=True) or {}
    lp = (j.get("layout_path") or "").strip()
    idx = int(j.get("index", -1))
    fs = j.get("fontSize"); col = j.get("color")
    if not lp or not pathlib.Path(lp).exists():
        return {"error":"layout not found"}, 400
    d = load_layout(lp)
    if not (0 <= idx < len(d["boxes"])): return {"error":"index out of range"}, 400
    while len(d["styles"]) < len(d["boxes"]):
        d["styles"].append({"fontSize": None, "color": None})
    if fs is not None: d["styles"][idx]["fontSize"] = int(fs)
    if col: d["styles"][idx]["color"] = col
    save_layout(lp, d)
    return {"ok":True}

@app.post("/api/render")
def api_render():
    j = request.get_json(force=True, silent=True) or {}
    lp = (j.get("layout_path") or "").strip()
    tr = j.get("translations")
    if not lp or not pathlib.Path(lp).exists():
        return {"error":"layout not found"}, 400
    d = load_layout(lp)
    if not isinstance(tr, list) or not tr:
        tr = d.get("translations", [])
    if len(tr) != len(d["boxes"]):
        tr = (tr + [""]*len(d["boxes"]))[:len(d["boxes"])]
    clean_path = _ensure_clean(lp, d, force=False)
    font_path = DEFAULT_FONT_PATH
    _, _, edited_img, _ = _paths(lp, d["source_image"])
    try:
        img = render_translations_styled(clean_path, d["boxes"], tr, d.get("styles", []), font_path)
    except Exception:
        img = render_translations(d["source_image"], d["boxes"], tr, font_path)
    img.save(edited_img, "PNG", optimize=False)
    d["translations"] = tr
    save_layout(lp, d)
    return {"ok":True, "out_image": str(edited_img)}

# ---------- move/resize boxes ----------
@app.post("/api/update_boxes")
def api_update_boxes():
    j = request.get_json(force=True, silent=True) or {}
    lp = (j.get("layout_path") or "").strip()
    updates = j.get("boxes") or []
    if not lp or not pathlib.Path(lp).exists():
        return {"error":"layout not found"}, 400
    d = load_layout(lp)
    n = len(d["boxes"])
    changed=0
    for it in updates:
        try:
            i = int(it.get("index")); bb = it.get("bbox") or []
            if 0 <= i < n and len(bb)==4:
                x,y,w,h = [int(v) for v in bb]
                d["boxes"][i].bbox = (x,y,w,h); changed += 1
        except Exception:
            pass
    save_layout(lp, d)
    return {"ok":True, "changed": changed}

# ---------- list / download ----------
@app.get("/api/list")
def api_list():
    items=[]
    for p in OUT_DIR.glob("*.layout.json"):
        try:
            j = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
            src = j.get("source_image"); stem = pathlib.Path(src).stem
            items.append({
                "layout": str(p),
                "source": str(src),
                "clean": str(OUT_DIR / f"{stem}.clean.png"),
                "edited": str(OUT_DIR / f"{stem}.edited.png")
            })
        except Exception:
            continue
    return {"items": items}

@app.get("/api/download_zip")
def api_download_zip():
    try:
        include = (request.args.get("include") or "both").lower()
        do_wm = request.args.get("wm","0") == "1"
        raw = request.args.get("items","")
        if raw:
            paths = [pathlib.Path(x) for x in json.loads(raw)]
        else:
            paths = list(OUT_DIR.glob("*.layout.json"))
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
            for lp in paths:
                try:
                    d = load_layout(str(lp))
                    clean_path = _ensure_clean(str(lp), d, force=False)
                    font_path = DEFAULT_FONT_PATH
                    try:
                        img = render_translations_styled(clean_path, d["boxes"], d.get("translations", []), d.get("styles", []), font_path)
                    except Exception:
                        img = render_translations(d["source_image"], d["boxes"], d.get("translations", []), font_path)
                    if do_wm:
                        img = apply_watermark(img)
                    stem = pathlib.Path(d["source_image"]).stem
                    edited_p = OUT_DIR / f"{stem}.edited.png"
                    img.save(edited_p, "PNG", optimize=False)

                    z.writestr(f"layouts/{lp.name}", json.dumps({
                        "source_image": d["source_image"],
                        "boxes": [{"text": b.text, "bbox": list(b.bbox)} for b in d["boxes"]],
                        "translations": d.get("translations", []),
                        "styles": d.get("styles", []),
                        "erasers": d.get("erasers", []),
                    }, ensure_ascii=False, indent=2))
                    cp = OUT_DIR / f"{stem}.clean.png"
                    if include in ("both","clean") and cp.exists(): z.write(cp, f"images/{stem}.clean.png")
                    if include in ("both","edited") and edited_p.exists(): z.write(edited_p, f"images/{stem}.edited.png")
                except Exception:
                    continue
        mem.seek(0)
        return send_file(mem, mimetype="application/zip", as_attachment=True, download_name="translated_images.zip")
    except Exception as ex:
        return {"error": str(ex)}, 500

# ---------- HTML ----------
HOME_HTML = r"""<!doctype html>
<html>
<head><meta charset="utf-8"/>
<title>Translator</title>
<style>
  :root{--bg:#e9eff1;--panel:#fff;--accent:#111827;--muted:#6b7280}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);font-family:ui-sans-serif,system-ui,Segoe UI,Arial}
  .hero{padding:46px 12px;text-align:center}
  h1{font-size:44px;margin:0 0 16px}
  .card{width:min(920px,96vw);margin:12px auto;background:var(--panel);border:1px solid #e5e7eb;border-radius:16px;padding:28px}
  .drop{border:2px dashed #cbd5e1;border-radius:14px;display:flex;align-items:center;justify-content:center;height:260px;color:#64748b}
  input[type=file]{display:none}
  .btn{background:#111827;color:#fff;border:1px solid #111827;border-radius:10px;padding:10px 14px;cursor:pointer}
  .row{display:flex;gap:10px;align-items:center;justify-content:center;margin-top:12px}
  select{padding:8px;border-radius:10px;border:1px solid #d1d5db}
  .hint{color:var(--muted);font-size:12px;margin-top:6px}
</style>
</head>
<body>
  <div class="hero">
    <h1>Translate multiple images to any language</h1>
    <div class="card">
      <label class="drop">
        <input id="files" type="file" multiple accept=".png,.jpg,.jpeg,.webp,.bmp"/>
        <div id="dropText">Drag & Drop , Paste or Click to upload</div>
      </label>
      <div class="row">
        <label>Target language</label>
        <select id="manga"><option value="1" selected>Manga mode</option><option value="0">Normal</option></select>
        <button class="btn" id="go">Translate</button>
      </div>
      <div class="hint" id="status"></div>
    </div>
  </div>
<script>
const filesEl = document.getElementById('files');
const statusEl = document.getElementById('status');
let picked=[];

filesEl.addEventListener('change', ()=>{
  picked=[...filesEl.files];
  document.getElementById('dropText').textContent = picked.length
    ? picked.length+' image(s) selected'
    : 'Drag & Drop , Paste or Click to upload';
});

document.addEventListener('dragover', e=>e.preventDefault());
document.addEventListener('drop', e=>{
  e.preventDefault();
  const f=[...e.dataTransfer.files].filter(x=>/\.(png|jpe?g|webp|bmp)$/i.test(x.name));
  if(f.length){
    picked=f;
    const dt=new DataTransfer();
    f.forEach(x=>dt.items.add(x));
    filesEl.files=dt.files;
    document.getElementById('dropText').textContent=f.length+' image(s) selected';
  }
});

document.getElementById('go').addEventListener('click', start);

async function start(){
  if(!picked.length){ toast('Select images'); return; }
  try{
    toast('Uploading…');
    const fd=new FormData(); picked.forEach(f=>fd.append('files',f,f.name));
    const r=await fetch('/api/upload',{method:'POST',body:fd});
    const j=await r.json();
    if(!r.ok || !j.ok){ toast(j.error||'Upload failed',true); return; }

    toast('Translating…');
    const r2=await fetch('/api/auto_translate',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({ images:j.files, manga_mode:document.getElementById('manga').value==='1' })
    });
    const j2=await r2.json();
    if(!r2.ok || !j2.results){ toast(j2.error||'Translate failed',true); return; }

    const okRes = j2.results.find(x => x && x.layout);
    if(!okRes){
      const firstErr = j2.results.find(x => x && x.error);
      toast(firstErr ? ('Failed: '+firstErr.error) : 'No layouts produced', true);
      return;
    }
    location.href = '/editor?layout=' + encodeURIComponent(okRes.layout);
  }catch(ex){
    toast('Unexpected error: '+ex, true);
  }
}

function toast(m,err=false){
  statusEl.style.color = err ? '#ef4444' : '#64748b';
  statusEl.textContent = m;
}
</script>

</body></html>
"""

EDITOR_HTML = r"""<!doctype html>
<html>
<head><meta charset="utf-8"/>
<title>Editor</title>
<style>
  :root{--bg:#0f1115;--panel:#ffffff;--border:#e5e7eb;--muted:#6b7280}
  *{box-sizing:border-box}

  body{margin:0;height:100vh;display:grid;grid-template-columns:1fr 420px;background:var(--bg);color:#111;font-family:ui-sans-serif,system-ui,Segoe UI,Arial}
  #left{position:relative;overflow:auto}
  #right{background:var(--panel);padding:14px;overflow:auto;border-left:1px solid var(--border)}
  .hdr{position:sticky;top:0;background:#fff;padding:10px 0;border-bottom:1px solid #e5e7eb;font-weight:700;font-size:18px;z-index:3}
  #stage{position:relative;display:inline-block;margin:14px}
  #img{display:block;max-width:none}
  #brush{position:absolute;left:0;top:0;pointer-events:none}
  #ui{position:fixed;left:16px;top:16px;display:flex;gap:8px;z-index:5}
  .btn{padding:8px 10px;border-radius:10px;border:1px solid #d1d5db;background:#0f172a;color:#fff;cursor:pointer}
  .chip{padding:6px 10px;border:1px solid var(--border);border-radius:999px;background:#f8fafc;color:#111;cursor:pointer}
  .chip.active{background:#0f172a;color:#fff;border-color:#0f172a}
  input[type=color],select{border:1px solid var(--border);border-radius:8px;padding:6px}
  .row{display:flex;gap:8px;align-items:center;margin:8px 0;flex-wrap:wrap}
  .ok{color:#0a7a2f}.err{color:#b00020;white-space:pre-wrap}

  /* Overlay box (нэг л элементийг чирж/жижгэрүүлнэ) */
  .box{
    position:absolute;
    border:2px solid rgba(0,200,255,.9);
    border-radius:8px;
    cursor:move;
    user-select:none;
  }
  .box.active{ outline: 2px dashed #0ea5e9; }
  .box .txt{
    position:absolute;
    left:4px; top:4px; right:4px; bottom:4px;
    overflow:hidden;
    white-space:pre-wrap;
    line-height:1.15;
    color:#111;
    cursor:text;
  }
  .box .handle{
    position:absolute; right:-6px; bottom:-6px;
    width:12px; height:12px; border-radius:3px;
    background:#0f172a; cursor:nwse-resize;
  }

  /* Sidebar list */
  .item{border:1px solid #e5e7eb;border-radius:12px;padding:10px;margin:8px 0}
  .item.active{outline:2px solid #0ea5e9}
  .item .o{font-size:12px;color:#6b7280;white-space:pre-wrap}
  .item textarea{width:100%;min-height:64px;border:1px solid #e5e7eb;border-radius:8px;padding:8px;font:inherit}
  .tools{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
</style>
</head>
<body>
  <div id="left">
    <div id="ui">
      <button class="btn" onclick="fit()">Fit</button>
      <button class="btn" onclick="actual()">1:1</button>
      <button class="btn" onclick="undo()" title="Ctrl+Z">Undo</button>
      <button class="btn" onclick="redo()" title="Ctrl+Y">Redo</button>
      <span class="chip" id="vEd" onclick="setView('edited')">Edited (live)</span>
      <span class="chip" id="vOr" onclick="setView('original')">Original</span>
    </div>
    <div id="stage">
      <img id="img" src="">
      <canvas id="brush"></canvas>
    </div>
  </div>

  <div id="right">
    <div class="hdr">Lines (OCR → Translate → Edit)</div>

    <div class="row">
      <input id="layout" style="flex:1;padding:8px;border:1px solid #e5e7eb;border-radius:10px" placeholder="output/01.layout.json">
      <button class="btn" onclick="openLayout()">Load</button>
    </div>

    <div class="row tools">
      <span>Font</span>
      <select id="fsize"><option>16</option><option selected>24</option><option>32</option><option>40</option><option>48</option><option>56</option></select>
      <input type="color" id="fcol" value="#111111">
      <button class="btn" onclick="applyStyle()">Apply style</button>

      <button class="btn" onclick="saveBoxes()">Save boxes</button>
      <button class="btn" onclick="rebuild()">Rebuild clean</button>

      <button class="btn" onclick="renderAndDownload()">Download ZIP</button>
    </div>

    <div id="list"></div>
    <div id="stat"></div>

    <div class="hdr">AI Inpaint</div>
    <div class="row" id="brushRow">
      <span style="min-width:90px">Brush</span>
      <input type="range" id="bsize" min="8" max="200" value="90" style="flex:1">
      <button class="btn" onclick="setMode('paint')">Paint</button>
      <button class="btn" onclick="applyMask()">✓ Apply</button>
      <button class="btn" onclick="clearMask()">× Clear</button>
    </div>
  </div>

<script>
let layout=null, lp=null, clean='', edited='', view='edited', mode='paint';
let img=document.getElementById('img'), brush=document.getElementById('brush'), bctx=brush.getContext('2d');
let scale=1, boxes=[], active=-1, painting=false;
let history=[], future=[];
let dragging=false, resizing=false, dx=0, dy=0;

document.addEventListener('DOMContentLoaded', ()=>{
  const q=new URL(location.href).searchParams.get('layout');
  if(q){ document.getElementById('layout').value=q; openLayout(); }
  setView('edited');

  document.getElementById('left').addEventListener('wheel',e=>{ if(e.ctrlKey){ e.preventDefault(); stepZoom(e.deltaY<0?0.1:-0.1);}},{passive:false});
  brush.addEventListener('mousedown',startPaint); window.addEventListener('mousemove',movePaint); window.addEventListener('mouseup',endPaint);

  window.addEventListener('keydown',e=>{
    if(e.ctrlKey && e.key.toLowerCase()==='z'){ e.preventDefault(); undo(); }
    if(e.ctrlKey && e.key.toLowerCase()==='y'){ e.preventDefault(); redo(); }
    // Arrow move for active box
    if(active>=0 && ['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)){
      e.preventDefault();
      const b=boxes[active]; if(!b) return;
      const step = e.shiftKey?10:1;
      let x=parseInt(b.style.left)||0, y=parseInt(b.style.top)||0;
      if(e.key==='ArrowLeft')  x-=step;
      if(e.key==='ArrowRight') x+=step;
      if(e.key==='ArrowUp')    y-=step;
      if(e.key==='ArrowDown')  y+=step;
      b.style.left=x+'px'; b.style.top=y+'px';
      setActiveBox(active);
    }
    // Duplicate
    if(active>=0 && e.ctrlKey && (e.key==='d' || e.key==='D')){
      e.preventDefault();
      const src=boxes[active];
      const clone=src.cloneNode(true);
      clone.style.left=(parseInt(src.style.left)+12)+'px';
      clone.style.top =(parseInt(src.style.top )+12)+'px';
      document.getElementById('stage').appendChild(clone);
      boxes.push(clone);
      layout.boxes.push({text:'', bbox:[
        parseInt(clone.style.left),parseInt(clone.style.top),
        parseInt(clone.style.width),parseInt(clone.style.height)
      ]});
      setActiveBox(boxes.length-1);
    }
    // Delete
    if(active>=0 && e.key==='Delete'){
      e.preventDefault();
      boxes[active].remove();
      boxes.splice(active,1);
      layout.boxes.splice(active,1);
      active=Math.min(active, boxes.length-1);
      setActiveBox(active);
    }
  });
});

function setView(v){
  view=v;
  document.getElementById('vEd').classList.toggle('active',v==='edited');
  document.getElementById('vOr').classList.toggle('active',v==='original');
  refreshImage(true);
}
function setMode(m){ mode=m; brush.style.pointerEvents = (mode==='paint' && view==='edited')?'auto':'none'; }
function stepZoom(d){ scale=Math.min(6,Math.max(0.1,scale+d)); document.getElementById('stage').style.transform='scale('+scale+')'; }
function actual(){ scale=1; document.getElementById('stage').style.transform='scale(1)'; }
function fit(){
  if(!img.naturalWidth) return;
  const wrap=document.getElementById('left'); const pad=60;
  const a=(wrap.clientWidth-pad)/img.naturalWidth, b=(wrap.clientHeight-pad)/img.naturalHeight;
  scale=Math.min(a,b); document.getElementById('stage').style.transform='scale('+scale+')';
}

async function openLayout(){
  const path=document.getElementById('layout').value.trim(); if(!path) return;
  const r=await fetch('/api/open?layout='+encodeURIComponent(path)); const j=await r.json();
  if(!r.ok){ stat(j.error,true); return; }
  layout=j; lp=path; clean=j.clean_image; edited=j.edited_image;
  await refreshImage(true);
  buildOverlays();
  buildList();
  clearHistory(); pushState();
  stat('OK: '+path);
}

async function refreshImage(reset=false){
  if(!layout) return;
  const url = (view==='original')
    ? '/api/img?path='+encodeURIComponent(layout.source_image)
    : '/api/img?path='+encodeURIComponent(clean);
  await new Promise(res=>{
    const tmp=new Image();
    tmp.onload=()=>{ img.src=url+'&t='+(Date.now()); img.onload=()=>{ sizeBrush(); if(reset) fit(); res(); }; };
    tmp.src=url+'&probe=1&t='+(Date.now());
  });
}
function sizeBrush(){ brush.width=img.naturalWidth; brush.height=img.naturalHeight; brush.style.width=img.naturalWidth+'px'; brush.style.height=img.naturalHeight+'px'; clearMask(); }

/* ======= Overlays ======= */
function buildOverlays(){
  boxes.forEach(x=>x.remove()); boxes=[];
  const parent=document.getElementById('stage');

  (layout.boxes||[]).forEach((b,i)=>{
    const [x,y,w,h]=b.bbox.map(v=>parseInt(v));
    const box=document.createElement('div'); box.className='box';
    box.style.left=x+'px'; box.style.top=y+'px';
    box.style.width=w+'px'; box.style.height=h+'px';

    const txt=document.createElement('div'); txt.className='txt';
    txt.contentEditable='true';
    const val=(layout.translations&&layout.translations[i])?layout.translations[i]:'';
    txt.textContent=val;

    const style=(layout.styles&&layout.styles[i])?layout.styles[i]:{};
    txt.style.fontSize=(style.fontSize||24)+'px';
    txt.style.color=(style.color||'#111111');

    const hdl=document.createElement('div'); hdl.className='handle';
    box.appendChild(txt); box.appendChild(hdl);
    parent.appendChild(box);

    box.addEventListener('mousedown',e=>{
      active=i; setActiveBox(i);
      const r=box.getBoundingClientRect();
      if(e.target===hdl){ resizing=true; }
      else { dragging=true; dx=e.clientX-r.left; dy=e.clientY-r.top; }
      document.body.style.userSelect='none';
    });
    window.addEventListener('mousemove',e=>{
      if(active!==i) return;
      if(dragging){
        const stg=brush.getBoundingClientRect();
        const nx=(e.clientX-stg.left)/scale - dx;
        const ny=(e.clientY-stg.top )/scale - dy;
        box.style.left=Math.round(nx)+'px';
        box.style.top =Math.round(ny)+'px';
      }else if(resizing){
        const bcr=box.getBoundingClientRect();
        const w=(e.clientX-bcr.left)/scale, h=(e.clientY-bcr.top)/scale;
        box.style.width = Math.max(30, Math.round(w))+'px';
        box.style.height= Math.max(20, Math.round(h))+'px';
      }
    });
    window.addEventListener('mouseup',()=>{ dragging=false; resizing=false; });

    box.addEventListener('dblclick', ()=>{ txt.focus(); });
    boxes.push(box);
  });
}
function setActiveBox(i){
  boxes.forEach((b,idx)=>b.classList.toggle('active', idx===i));
}

/* --------- Collectors --------- */
function collectBoxes(){
  return boxes.map((b,i)=>({
    index:i,
    bbox:[
      Math.round(parseInt(b.style.left)||0),
      Math.round(parseInt(b.style.top)||0),
      Math.round(parseInt(b.style.width)||0),
      Math.round(parseInt(b.style.height)||0),
    ]
  }));
}

/* ======= Sidebar ======= */
function buildList(){
  const host=document.getElementById('list'); host.innerHTML='';
  (layout.boxes||[]).forEach((b,i)=>{
    const it=document.createElement('div'); it.className='item'; it.id='it'+i;
    const o=document.createElement('div'); o.className='o'; o.textContent=b.text||'';
    const ta=document.createElement('textarea'); ta.value=(layout.translations&&layout.translations[i])?layout.translations[i]:'';
    ta.addEventListener('input',()=>{
      const el=boxes[i]?.querySelector('.txt'); if(el) el.textContent=ta.value;
    });
    ta.addEventListener('change',()=>{ pushState(); });
    it.addEventListener('click',()=>{ selectItem(i); });
    it.appendChild(o); it.appendChild(ta); host.appendChild(it);
  });
}
function selectItem(i){
  [...document.querySelectorAll('.item')].forEach((el,idx)=>el.classList.toggle('active',idx===i));
  active=i; boxes[i]?.scrollIntoView({block:'center',behavior:'smooth'});
}
function collectTranslations(){
  return [...document.querySelectorAll('#list textarea')].map(t=>t.value);
}

/* ======= Undo / Redo ======= */
function snapshot(){
  return {
    tr: collectTranslations(),
    bx: collectBoxes(),
    st: layout.styles ? JSON.parse(JSON.stringify(layout.styles)) : []
  };
}
function applySnapshot(s){
  const ta = [...document.querySelectorAll('#list textarea')];
  s.tr.forEach((v,i)=>{ if(ta[i]) ta[i].value=v; const el=boxes[i]?.querySelector('.txt'); if(el) el.textContent=v; });
  s.bx.forEach((b,i)=>{
    const box=boxes[i]; if(!box) return;
    const [x,y,w,h]=b.bbox;
    box.style.left=x+'px'; box.style.top=y+'px'; box.style.width=w+'px'; box.style.height=h+'px';
  });
  layout.styles = s.st;
}
function pushState(){ history.push(snapshot()); if(history.length>100) history.shift(); future.length=0; }
function clearHistory(){ history.length=0; future.length=0; }
function undo(){ if(history.length<=1) return; const cur=history.pop(); future.push(cur); applySnapshot(history[history.length-1]); }
function redo(){ if(future.length===0) return; const s=future.pop(); history.push(s); applySnapshot(s); }

/* ======= Actions ======= */
async function saveBoxes(){
  if(!lp) return;
  const r=await fetch('/api/update_boxes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({layout_path:lp, boxes:collectBoxes()})});
  const j=await r.json(); if(!r.ok){ stat(j.error,true); return; }
  stat('Boxes saved: '+j.changed);
}
async function applyStyle(){
  const i = Math.max(0, active);
  const fs=parseInt(document.getElementById('fsize').value,10);
  const col=document.getElementById('fcol').value;
  const b=boxes[i]; if(!b) return;
  const txt=b.querySelector('.txt');
  txt.style.fontSize=fs+'px'; txt.style.color=col;
  const r=await fetch('/api/update_style',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({layout_path:lp,index:i,fontSize:fs,color:col})});
  const j=await r.json(); if(!r.ok){ stat(j.error,true); return; }
  pushState(); stat('Style updated');
}
async function renderAndDownload(){
  if(!lp) return;
  await fetch('/api/update_boxes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({layout_path:lp, boxes:collectBoxes()})});
  const r=await fetch('/api/render',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({layout_path:lp, translations:collectTranslations()})});
  const j=await r.json(); if(!r.ok){ stat(j.error,true); return; }
  edited=j.out_image;
  const items=encodeURIComponent(JSON.stringify([lp]));
  location.href='/api/download_zip?include=edited&items='+items;
  stat('Rendered & downloading ZIP…');
}
async function rebuild(){
  if(!lp) return;
  const r=await fetch('/api/rebuild_clean',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({layout_path:lp})});
  const j=await r.json(); if(!r.ok){ stat(j.error,true); return; }
  clean=j.clean_image; await refreshImage(false); stat('Clean updated');
}
function stat(m,err=false){ document.getElementById('stat').innerHTML=err?('<span class="err">'+m+'</span>'):('<span class="ok">'+m+'</span>'); }
async function exists(p){ if(!p) return false; const r=await fetch('/api/exists?path='+encodeURIComponent(p)); if(!r.ok) return false; const j=await r.json(); return !!j.exists; }

/* ======= Brush Inpaint ======= */
function startPaint(e){
  if(mode!=='paint' || view!=='edited') return;
  painting=true; brush.style.pointerEvents='auto';
  bctx.globalCompositeOperation='source-over';
  bctx.strokeStyle='rgba(0,128,255,0.45)';
  bctx.lineWidth=+document.getElementById('bsize').value;
  bctx.lineCap='round'; bctx.lineJoin='round';
  const p=pt(e); bctx.beginPath(); bctx.moveTo(p.x,p.y);
}
function movePaint(e){
  if(!painting || mode!=='paint' || view!=='edited') return;
  const p=pt(e); bctx.lineTo(p.x,p.y); bctx.stroke();
}
function endPaint(){ painting=false; }
function pt(e){ const r=brush.getBoundingClientRect(); return {x:(e.clientX-r.left)/scale, y:(e.clientY-r.top)/scale}; }
function clearMask(){ bctx.clearRect(0,0,brush.width,brush.height); }
async function applyMask(){
  if(!lp) return;
  const tmp=document.createElement('canvas'); tmp.width=brush.width; tmp.height=brush.height;
  const t=tmp.getContext('2d'); t.fillStyle='black'; t.fillRect(0,0,tmp.width,tmp.height);
  t.globalCompositeOperation='source-over'; t.drawImage(brush,0,0);
  const dataURL = tmp.toDataURL('image/png');
  clearMask();
  const r=await fetch('/api/erase_mask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({layout_path:lp,mask_data_url:dataURL})});
  const j=await r.json(); if(!r.ok){ stat(j.error,true); return; }
  clean=j.clean_image; await refreshImage(false); stat('Inpaint OK');
}
</script>
</body></html>
"""

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    print(f"Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)

if __name__ == "__main__":
    main()
