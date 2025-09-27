# app.py
from __future__ import annotations
import argparse, io, os, json, base64, zipfile, pathlib, traceback, re as _re
from typing import List, Dict, Any, Tuple
from flask import Flask, request, Response, send_file, send_from_directory
from PIL import Image, ImageFilter  # ⬅️ MaxFilter/Blur ашиглана
import numpy as np

# ----- project deps -----
from utils import OCRBox, erase_text_area, inpaint_with_mask
from renderer import render_translations_styled, render_translations
from ocr_torii import ocr_image
from translate_lingva import translate_lines, translate_lines_impl
from config import DEFAULT_FONT_PATH

app = Flask(__name__, static_url_path="/static")
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024  # 256MB
IN_DIR  = pathlib.Path("input");  IN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = pathlib.Path("output"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- watermark (optional) ----------
WATERMARK_PATH = os.environ.get("WATERMARK_PATH", r"C:\FEDoUP\watermark.png")
WM_RATIO   = float(os.environ.get("WM_RATIO", "0.22"))
WM_MIN_W   = int(os.environ.get("WM_MIN_W", "100"))
WM_OFFSET  = os.environ.get("WM_OFFSET", "+12+12")
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
    j["restores"] = j.get("restores", [])  # <-- restore masks
    return j

def save_layout(p: str, d: Dict[str, Any]) -> None:
    out = {
        "source_image": d["source_image"],
        "boxes": [{"text": b.text, "bbox": list(b.bbox)} for b in d["boxes"]],
        "translations": d.get("translations", []),
        "styles": d.get("styles", []),
        "erasers": d.get("erasers", []),
        "restores": d.get("restores", []),
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def _ensure_clean(layout_path: str, data: Dict[str, Any], force=False) -> str:
    src_path, clean, _, _ = _paths(layout_path, data["source_image"])
    if clean.exists() and not force:
        return str(clean)

    original = Image.open(src_path).convert("RGBA")
    base = original.copy()

    # ---- ERASE (auto + mask) ----
    auto_erase = os.getenv("AUTO_ERASE", "1") not in ("0","false","False","no","No")

    # 1) ЭХЛЭЭД үргэлж auto pass (асуудалгүй идемпотент)
    if auto_erase:
        for b in data.get("boxes", []):
            try:
                erase_text_area(
                    base,
                    box=tuple(map(int, b.bbox)),
                    expand=18, feather=3, corner=8, aggressive=3,
                    mask_mode="text", pattern=None
                )
            except Exception:
                pass

    # 2) ДАРАА НЬ таны гарын erasers (bbox/mask)
    for e in data.get("erasers", []):
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
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
                out = inpaint_with_mask(base, mu8, (int(x1),int(y1),int(x2-x1),int(y2-y1)))
                if out is not None:
                    s = float(e.get("strength", 1.0))
                    if s >= 0.999:
                        base.paste(out)
                    else:
                        m = mask.point(lambda p: int(p * s))
                        base = Image.composite(out.convert("RGBA"), base, m)


    # ---- RESTORE (overlay original by mask with strength) ----
    for r in data.get("restores", []):
        if r.get("type") != "mask":
            continue
        mf = r.get("mask_file")
        if not mf or not pathlib.Path(mf).exists():
            continue
        m = Image.open(mf).convert("L")
        if m.size != original.size:
            m = m.resize(original.size, Image.NEAREST)
        s = float(r.get("strength", 1.0))
        if s < 0.999:
            m = m.point(lambda p: int(p * s))
        base = Image.composite(original, base, m)

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

# --- Box utils for grouping + pruning ---
_NOISE_PUNCT_RE = _re.compile(r"^[\s\.\,\!\?\-—–…:;~·・/\\|]+$")
_UP_ASCII_RE    = _re.compile(r"^[A-Z0-9\s'!?\.]+$")

def _is_noise_text(t: str) -> bool:
    s = (t or "").strip()
    if not s: return True
    if _NOISE_PUNCT_RE.match(s): return True
    if len(s) <= 1: return True
    if len(s) <= 12 and _UP_ASCII_RE.fullmatch(s) and s == s.upper() and s not in ("OK","TV"):
        return True
    return False

def _rect_union(a, b):
    ax,ay,aw,ah = a; bx,by,bw,bh = b
    x1=min(ax,bx); y1=min(ay,by)
    x2=max(ax+aw,bx+bw); y2=max(ay+ah,bx+bh)
    return (x1,y1,x2-x1,y2-y1)

def _rect_inflate(a, d):
    x,y,w,h=a
    return (x-d, y-d, w+2*d, h+2*d)

def _rect_intersects(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    return not (ax+aw<=bx or bx+bw<=ax or ay+ah<=by or by+bh<=ay)

def _group_ocr_boxes(boxes: List[OCRBox], gap: int = 12):
    """Return list of tuples: (group_bbox, member_indices)"""
    n=len(boxes); used=[False]*n; groups=[]
    for i in range(n):
        if used[i]: continue
        rect = boxes[i].bbox
        members=[i]; used[i]=True
        changed=True
        while changed:
            changed=False
            infl=_rect_inflate(rect,gap)
            for j in range(n):
                if used[j]: continue
                if _rect_intersects(infl, _rect_inflate(boxes[j].bbox, gap)):
                    rect=_rect_union(rect, boxes[j].bbox)
                    members.append(j); used[j]=True; changed=True
        members = sorted(members, key=lambda k:(boxes[k].bbox[1], boxes[k].bbox[0]))
        groups.append((rect, members))
    return groups

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

# ---------- auto translate (batch) ----------
@app.post("/api/auto_translate")
def api_auto_translate():
    j = request.get_json(force=True, silent=True) or {}
    images: List[str] = j.get("images") or []
    manga_mode = bool(j.get("manga_mode", True))
    max_chars = int(j.get("max_chars", 800))
    api_key = os.getenv("TORII_API_KEY")

    results = []
    for ip in images:
        try:
            src = pathlib.Path(ip)
            if not src.exists():
                results.append({"image": ip, "error": "not found"})
                continue

            with Image.open(src) as im:
                _ = im.size

            # OCR → raw line boxes
            ocr = ocr_image(str(src), api_key)
            line_boxes = _extract_boxes(ocr)

            # ---- 1) Group lines by bubble ----
            groups = _group_ocr_boxes(line_boxes, gap=12)

            group_rects, group_src_texts = [], []
            for rect, idxs in groups:
                joined = " ".join((line_boxes[k].text or "").strip() for k in idxs).strip()
                if _is_noise_text(joined):
                    continue
                group_rects.append(rect)
                group_src_texts.append(joined[:max_chars])

            # If nothing useful – still produce empty layout
            if not group_src_texts:
                out_json = OUT_DIR / f"{src.stem}.layout.json"
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump({
                        "source_image": str(src),
                        "boxes": [],
                        "translations": [],
                        "styles": [],
                        "erasers": [],
                        "restores": [],
                    }, f, ensure_ascii=False, indent=2)
                results.append({"image": src.as_posix(), "layout": out_json.as_posix(),
                                "clean": "", "edited": ""})
                continue

            # ---- 2) Translate (1 group = 1 box) ----
            if manga_mode:
                trans = translate_lines_impl(group_src_texts, reflow=False)
            else:
                try:
                    trans = translate_lines(group_src_texts, reflow=False)
                except Exception:
                    trans = translate_lines_impl(group_src_texts, reflow=False)

            # ---- 3) Drop groups with empty/punct-only translations ----
            def _is_empty_tr(s: str) -> bool:
                s = (s or "").strip()
                return (not s) or _NOISE_PUNCT_RE.match(s)

            filt_rects, filt_srcs, filt_trans = [], [], []
            for r, s, t in zip(group_rects, group_src_texts, trans):
                if _is_empty_tr(t):
                    continue
                filt_rects.append(r); filt_srcs.append(s); filt_trans.append(t)

            # ---- 4) Write layout ----
            out_json = OUT_DIR / f"{src.stem}.layout.json"
            boxes_for_layout = [{"text": s, "bbox": list(r)} for s, r in zip(filt_srcs, filt_rects)]
            styles = [{"fontSize": None, "color": None} for _ in boxes_for_layout]

            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({
                    "source_image": str(src),
                    "boxes": boxes_for_layout,
                    "translations": filt_trans,
                    "styles": styles,
                    "erasers": [],
                    "restores": [],
                }, f, ensure_ascii=False, indent=2)

            # ---- 5) Clean + render ----
            clean_path = _ensure_clean(str(out_json), load_layout(str(out_json)), force=True)
            font_path = DEFAULT_FONT_PATH
            try:
                boxes_objs = [OCRBox(b["text"], tuple(b["bbox"])) for b in boxes_for_layout]
                img = render_translations_styled(clean_path, boxes_objs, filt_trans, styles, font_path)
            except Exception:
                img = render_translations(str(src), boxes_objs, filt_trans, font_path)

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

    return {"ok": True, "results": results}

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
        "restores": d.get("restores", []),
        "clean_image": str(clean),
        "edited_image": str(edited),
    }

# ---------- erase / restore masks + counters ----------
@app.post("/api/set_eraser_count")
def api_set_eraser_count():
    j = request.get_json(force=True, silent=True) or {}
    lp = (j.get("layout_path") or "").strip()
    count = int(j.get("count", -1))
    if not lp or not pathlib.Path(lp).exists():
        return {"error":"layout not found"}, 400
    d = load_layout(lp)
    if count < 0 or count > len(d.get("erasers", [])):
        return {"error":"bad count"}, 400
    d["erasers"] = d.get("erasers", [])[:count]
    save_layout(lp, d)
    clean_path = _ensure_clean(lp, d, force=True)
    return {"ok": True, "clean_image": clean_path, "eraser_count": len(d["erasers"])}

@app.post("/api/set_restore_count")
def api_set_restore_count():
    j = request.get_json(force=True, silent=True) or {}
    lp = (j.get("layout_path") or "").strip()
    count = int(j.get("count", -1))
    if not lp or not pathlib.Path(lp).exists():
        return {"error":"layout not found"}, 400
    d = load_layout(lp)
    if count < 0 or count > len(d.get("restores", [])):
        return {"error":"bad count"}, 400
    d["restores"] = d.get("restores", [])[:count]
    save_layout(lp, d)
    clean_path = _ensure_clean(lp, d, force=True)
    return {"ok": True, "clean_image": clean_path, "restore_count": len(d["restores"])}

@app.post("/api/erase_mask")
def api_erase_mask():
    j = request.get_json(force=True, silent=True) or {}
    lp = (j.get("layout_path") or "").strip()
    data_url = j.get("mask_data_url") or ""
    strength = float(j.get("strength", 1.0))
    if not lp or not pathlib.Path(lp).exists():
        return {"error":"layout not found"}, 400
    if not data_url.startswith("data:image/png;base64,"):
        return {"error":"invalid mask"}, 400

    d = load_layout(lp)
    src, _, _, mask_dir = _paths(lp, d["source_image"])
    b64 = data_url.split(",",1)[1]
    # 1) grayscale
    mask_im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
    W0, H0 = Image.open(src).size
    if mask_im.size != (W0, H0):
        mask_im = mask_im.resize((W0, H0), Image.NEAREST)

    # 2) robust binary mask (threshold + optional dilate + feather)
    thr = int(os.getenv("ERASE_THRESHOLD", "8"))
    mask_bin = mask_im.point(lambda p: 255 if p > thr else 0)
    dil = int(os.getenv("ERASE_DILATE", "3"))
    if dil > 0:
        # kernel size = 2*dil+1 (odd)
        k = max(3, 2*dil+1)
        mask_bin = mask_bin.filter(ImageFilter.MaxFilter(k))
    feather = int(os.getenv("ERASE_FEATHER", "0"))
    if feather > 0:
        mask_bin = mask_bin.filter(ImageFilter.GaussianBlur(radius=feather))

    idx = sum(1 for e in d.get("erasers",[]) if e.get("type")=="mask") + 1
    mpath = mask_dir / f"mask_{idx:03d}.png"
    mask_bin.save(mpath, "PNG")
    d["erasers"].append({"type":"mask", "mask_file": str(mpath), "strength": strength})
    save_layout(lp, d)
    clean_path = _ensure_clean(lp, d, force=True)
    return {"ok":True, "clean_image": clean_path, "mask_file": str(mpath), "eraser_count": len(d["erasers"])}

@app.post("/api/restore_mask")
def api_restore_mask():
    j = request.get_json(force=True, silent=True) or {}
    lp = (j.get("layout_path") or "").strip()
    data_url = j.get("mask_data_url") or ""
    strength = float(j.get("strength", 1.0))
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

    # restore маск ч мөн бинар/feather
    thr = int(os.getenv("RESTORE_THRESHOLD", "8"))
    mask_bin = mask_im.point(lambda p: 255 if p > thr else 0)
    feather = int(os.getenv("RESTORE_FEATHER", "0"))
    if feather > 0:
        mask_bin = mask_bin.filter(ImageFilter.GaussianBlur(radius=feather))

    idx = sum(1 for e in d.get("restores",[]) if e.get("type")=="mask") + 1
    mpath = mask_dir / f"restore_{idx:03d}.png"
    mask_bin.save(mpath, "PNG")
    d.setdefault("restores", []).append({"type":"mask", "mask_file": str(mpath), "strength": strength})
    save_layout(lp, d)
    clean_path = _ensure_clean(lp, d, force=True)
    return {"ok":True, "clean_image": clean_path, "mask_file": str(mpath), "restore_count": len(d["restores"])}

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
                        "restores": d.get("restores", []),
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
        <label>Mode</label>
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

    const okLayouts = j2.results.filter(x => x && x.layout).map(x => x.layout);
    if(!okLayouts.length){
      const firstErr = j2.results.find(x => x && x.error);
      toast(firstErr ? ('Failed: '+firstErr.error) : 'No layouts produced', true);
      return;
    }
    const q = encodeURIComponent(JSON.stringify(okLayouts));
    location.href = '/editor?layouts=' + q; // batch → editor
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
  #right{background:var(--panel);padding:14px;overflow:auto;border-left:1px solid #e5e7eb}
  .hdr{position:sticky;top:0;background:#fff;padding:10px 0;border-bottom:1px solid #e5e7eb;font-weight:700;font-size:18px;z-index:3}
  #stage{position:relative;display:inline-block;margin:14px}
  #img{display:block;max-width:none}
  #brush{position:absolute;left:0;top:0;pointer-events:none}
  #overlay{position:absolute;left:0;top:0;pointer-events:none}
  #ui{position:fixed;left:16px;top:16px;display:flex;gap:8px;z-index:5;align-items:center}
  .btn{padding:8px 10px;border-radius:10px;border:1px solid #d1d5db;background:#0f172a;color:#fff;cursor:pointer}
  .chip{padding:6px 10px;border:1px solid var(--border);border-radius:999px;background:#f8fafc;color:#111;cursor:pointer}
  .chip.active{background:#0f172a;color:#fff;border-color:#0f172a}
  input[type=color],select{border:1px solid var(--border);border-radius:8px;padding:6px}
  .row{display:flex;gap:8px;align-items:center;margin:8px 0;flex-wrap:wrap}
  .ok{color:#0a7a2f}.err{color:#b00020;white-space:pre-wrap}
  .box{position:absolute;border:2px solid rgba(14,165,233,0);border-radius:8px;cursor:move;user-select:none;background:transparent;pointer-events:auto;transition:border-color .06s ease;}
  .box:hover{ border-color: rgba(14,165,233,.35); }
  .box.active{ border-color: rgba(14,165,233,1); outline: 2px dashed #0ea5e9; }
  .box .txt{
  position:absolute; left:6px; top:6px; right:6px; bottom:6px;
  padding:2px 4px; overflow:hidden;
  white-space:pre-wrap;
  word-break: break-word;
  overflow-wrap: anywhere;
  line-height:1.15; color:#111; cursor:text;
  }
  .box .handle{position:absolute; right:-6px; bottom:-6px;width:12px;height:12px;border-radius:3px;background:#0f172a; cursor:nwse-resize; display:none;}
  .box.active .handle{ display:block; }
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
      <button class="btn" onclick="prevItem()" title="Previous">◀</button>
      <select id="batchSelect" onchange="jumpTo(this.value)" style="max-width:260px"></select>
      <button class="btn" onclick="nextItem()" title="Next">▶</button>
    </div>
    <div id="stage">
      <img id="img" src="">
      <canvas id="brush"></canvas>
      <div id="overlay"></div>
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

      <button class="btn" onclick="autoFitAll()">Auto-fit all</button>
      <!-- NEW: add blank text box -->
      <button class="btn" onclick="addBox()">Add box</button>
      <button class="btn" onclick="renderAndDownload()">Download ZIP</button>
    </div>

    <div id="list"></div>
    <div id="stat"></div>

    <div class="hdr">AI Inpaint / Restore</div>
      <div class="row" id="brushRow">
        <span style="min-width:90px">Brush</span>
        <input type="range" id="bsize" min="8" max="200" value="90" style="flex:1">
        <label style="margin-left:8px">Strength
          <span id="strengthVal">100%</span>
        </label>
        <input type="range" id="strength" min="0" max="100" value="100" style="width:160px">
        <button class="btn" onclick="setMode('paint')">Erase</button>
        <button class="btn" onclick="setMode('restore')">Restore</button>
        <button class="btn" onclick="applyMask()">✓ Apply Erase</button>
        <button class="btn" onclick="applyRestore()">✓ Apply Restore</button>
        <button class="btn" onclick="clearMask()">× Clear</button>
      </div>
  </div>

<script>
let batch = []; let idx = 0;

let layout=null, lp=null, clean='', edited='', view='edited', mode='paint';
let img=document.getElementById('img'), brush=document.getElementById('brush'), bctx=brush.getContext('2d');
let overlay=document.getElementById('overlay');
let scale=1, boxes=[], active=-1, painting=false;
let history=[], future=[];
let dragging=false, resizing=false, dx=0, dy=0, changed=false;
let eraserCount=0, restoreCount=0;

const statusEl = document.getElementById('stat');
const strengthEl = document.getElementById('strength');
const strengthVal = document.getElementById('strengthVal');

document.addEventListener('DOMContentLoaded', ()=>{
  const url = new URL(location.href);
  const ls = url.searchParams.get('layouts');
  const single = url.searchParams.get('layout');

  if (ls) {
    try { batch = JSON.parse(ls); idx = 0; fillBatchSelect(); openLayoutByIndex(idx); } catch(e) { console.error(e); }
  } else if (single) {
    document.getElementById('layout').value = single; openLayout();
  }
  setView('edited');

  document.getElementById('left').addEventListener('wheel',e=>{ if(e.ctrlKey){ e.preventDefault(); stepZoom(e.deltaY<0?0.1:-0.1);}},{passive:false});
  brush.addEventListener('mousedown',startPaint); window.addEventListener('mousemove',movePaint); window.addEventListener('mouseup',endPaint);

  strengthEl.addEventListener('input', ()=>{ strengthVal.textContent = (strengthEl.value|0) + '%'; });

  window.addEventListener('keydown',e=>{
    if(e.key === 'Escape'){ setActiveBox(-1); return; }
    if(e.ctrlKey && e.key.toLowerCase()==='z'){ e.preventDefault(); undo(); }
    if(e.ctrlKey && e.key.toLowerCase()==='y'){ e.preventDefault(); redo(); }
    if(view!=='edited') return;

    if(active>=0 && ['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)){
      e.preventDefault();
      const b=boxes[active]; if(!b) return;
      const step = e.shiftKey?10:1;
      let x=parseInt(b.style.left)||0, y=parseInt(b.style.top)||0;
      const ox=x, oy=y;
      if(e.key==='ArrowLeft')  x-=step;
      if(e.key==='ArrowRight') x+=step;
      if(e.key==='ArrowUp')    y-=step;
      if(e.key==='ArrowDown')  y+=step;
      if(x!==ox || y!==oy){ b.style.left=x+'px'; b.style.top=y+'px'; changed=true; }
    }
    if(view==='edited' && active>=0 && e.ctrlKey && (e.key==='d' || e.key==='D')){
      e.preventDefault();
      const src=boxes[active];
      const clone=src.cloneNode(true);
      clone.style.left=(parseInt(src.style.left)+12)+'px';
      clone.style.top =(parseInt(src.style.top )+12)+'px';
      overlay.appendChild(clone);
      boxes.push(clone);
      layout.boxes.push({text:'', bbox:[
        parseInt(clone.style.left),parseInt(clone.style.top),
        parseInt(clone.style.width),parseInt(clone.style.height)
      ]});
      setActiveBox(boxes.length-1);
      pushState();
      buildList();
    }
    if(view==='edited' && active>=0 && e.key==='Delete'){
      e.preventDefault();
      boxes[active].remove();
      boxes.splice(active,1);
      layout.boxes.splice(active,1);
      layout.translations?.splice(active,1);
      layout.styles?.splice(active,1);
      active=Math.min(active, boxes.length-1);
      setActiveBox(active);
      pushState();
      buildList();
    }
  });

  overlay.addEventListener('mousedown',e=>{ if(e.target===overlay){ setActiveBox(-1); } });
  setMode('paint'); // default color
});

function fillBatchSelect(){
  const sel = document.getElementById('batchSelect');
  sel.innerHTML = '';
  batch.forEach((p,i)=>{
    const opt = document.createElement('option');
    const name = (p.split(/[\\/]/).pop() || ('Page '+(i+1)));
    opt.value = i; opt.textContent = `${i+1}/${batch.length} — ${name}`;
    sel.appendChild(opt);
  });
  sel.value = idx.toString();
}
async function openLayoutByIndex(i){
  if (i<0 || i>=batch.length) return;
  idx = i;
  document.getElementById('layout').value = batch[idx];
  await openLayout();
  const sel = document.getElementById('batchSelect');
  if (sel) sel.value = idx.toString();
}
function prevItem(){ if (idx>0) openLayoutByIndex(idx-1); }
function nextItem(){ if (idx<batch.length-1) openLayoutByIndex(idx+1); }
function jumpTo(v){ const i = parseInt(v,10); if(!isNaN(i)) openLayoutByIndex(i); }

function setView(v){
  view=v;
  document.getElementById('vEd').classList.toggle('active',v==='edited');
  document.getElementById('vOr').classList.toggle('active',v==='original');
  const showOverlay = (v==='edited');
  overlay.style.display = showOverlay ? 'block' : 'none';
  brush.style.display   = showOverlay ? 'block' : 'none';
  brush.style.pointerEvents = showOverlay ? 'auto' : 'none';
  refreshImage(true);
}

function setMode(m){
  mode=m;
  const on = (view==='edited');
  brush.style.pointerEvents = on ? 'auto' : 'none';
  if (mode==='paint') {
    bctx.strokeStyle='rgba(0,128,255,0.45)'; // erase = blue
  } else if (mode==='restore') {
    bctx.strokeStyle='rgba(0,200,0,0.45)';   // restore = green
  }
}

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
  eraserCount = (layout.erasers || []).length;
  restoreCount = (layout.restores || []).length;
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
function sizeBrush(){
  brush.width=img.naturalWidth; brush.height=img.naturalHeight;
  brush.style.width=img.naturalWidth+'px'; brush.style.height=img.naturalHeight+'px';
  overlay.style.width=img.naturalWidth+'px'; overlay.style.height=img.naturalHeight+'px';
  clearMask();
}

function ensureMinSize(box){
  const txt = box.querySelector('.txt');
  const fs = parseInt(window.getComputedStyle(txt).fontSize) || 24;
  const MIN_W = Math.max(40, Math.round(fs * 6));
  const MIN_H = Math.max(28, Math.round(fs * 2));
  const w0 = parseInt(box.dataset.w0) || 0;
  const curW = parseInt(box.style.width)  || 0;
  const curH = parseInt(box.style.height) || 0;
  box.style.width  = Math.max(MIN_W, w0, curW) + 'px';
  box.style.height = Math.max(MIN_H, curH) + 'px';
}

function autoFit(i){
  const box = boxes[i]; if(!box) return;
  const txt = box.querySelector('.txt');
  const pad = 16;

  const w0 = parseInt(box.dataset.w0) || parseInt(box.style.width) || 120;
  const maxGrow = Math.round(w0 * 1.35);

  box.style.width  = w0 + 'px';
  box.style.height = 'auto';

  const needW = Math.ceil(txt.scrollWidth + pad);
  if (needW > w0) {
    box.style.width = Math.min(needW, maxGrow) + 'px';
  }
  const needH = Math.ceil(txt.scrollHeight + pad);
  box.style.height = needH + 'px';

  ensureMinSize(box);
}
function autoFitAll(){ (boxes||[]).forEach((_,i)=>autoFit(i)); pushState(); stat('Auto-fit all done'); }

function buildOverlays(){
  boxes.forEach(x=>x.remove()); boxes=[];
  overlay.innerHTML='';
  (layout.boxes||[]).forEach((b,i)=>{
    const [x,y,w,h]=b.bbox.map(v=>parseInt(v));
    const box=document.createElement('div'); box.className='box';
    box.style.left=x+'px'; box.style.top=y+'px';
    box.style.width=w+'px'; box.style.height=h+'px';
    box.dataset.w0 = w; // ⬅️ OCR bbox өргөнийг хадгална
    const txt=document.createElement('div'); txt.className='txt'; txt.contentEditable='true';
    txt.textContent=(layout.translations&&layout.translations[i])?layout.translations[i]:'';
    const style=(layout.styles&&layout.styles[i])?layout.styles[i]:{};
    txt.style.fontSize=(style.fontSize||24)+'px'; txt.style.color=(style.color||'#111111');
    const hdl=document.createElement('div'); hdl.className='handle';
    box.appendChild(txt); box.appendChild(hdl); overlay.appendChild(box);
    ensureMinSize(box);
    requestAnimationFrame(()=> autoFit(i));

    box.addEventListener('mousedown',e=>{
      if(view!=='edited') return;
      active=i; setActiveBox(i);
      const r=box.getBoundingClientRect();
      changed=false;
      if(e.target===hdl){ resizing=true; } else { dragging=true; dx=e.clientX-r.left; dy=e.clientY-r.top; }
      document.body.style.userSelect='none';
    });
    window.addEventListener('mousemove',e=>{
      if(view!=='edited' || active!==i) return;
      if(dragging){
        const stg=brush.getBoundingClientRect();
        const nx=Math.round((e.clientX-stg.left)/scale - dx);
        const ny=Math.round((e.clientY-stg.top )/scale - dy);
        if(nx!==parseInt(box.style.left) || ny!==parseInt(box.style.top)){
          box.style.left=nx+'px'; box.style.top=ny+'px'; changed=true;
        }
      }else if(resizing){
        const bcr=box.getBoundingClientRect();
        const nw=Math.max(30, Math.round((e.clientX-bcr.left)/scale));
        const nh=Math.max(20, Math.round((e.clientY-bcr.top )/scale));
        if(nw!==parseInt(box.style.width) || nh!==parseInt(box.style.height)){
          box.style.width=nw+'px'; box.style.height=nh+'px'; changed=true;
        }
      }
    });
    window.addEventListener('mouseup',()=>{ if(dragging||resizing){ if(changed) pushState(); } dragging=false; resizing=false; changed=false; });
    box.addEventListener('dblclick', ()=>{ if(view==='edited'){ autoFit(i); pushState(); }});
    txt.addEventListener('input', ()=> autoFit(i));
    boxes.push(box);
  });
  setActiveBox(-1);
}
function setActiveBox(i){
  active=i;
  boxes.forEach((b,idx)=>b.classList.toggle('active', idx===i));
  [...document.querySelectorAll('.item')].forEach((el,idx)=>el.classList.toggle('active',idx===i));
}
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
function buildList(){
  const host=document.getElementById('list'); host.innerHTML='';
  (layout.boxes||[]).forEach((b,i)=>{
    const it=document.createElement('div'); it.className='item'; it.id='it'+i;
    const o=document.createElement('div'); o.className='o'; o.textContent=b.text||'';
    const ta=document.createElement('textarea'); ta.value=(layout.translations&&layout.translations[i])?layout.translations[i]:'';
    ta.addEventListener('input',()=>{ const el=boxes[i]?.querySelector('.txt'); if(el) el.textContent=ta.value; autoFit(i); });
    ta.addEventListener('change',()=>{ pushState(); });
    it.addEventListener('click',()=>{ setActiveBox(i); boxes[i]?.scrollIntoView({block:'center',behavior:'smooth'}); });
    host.appendChild(it); it.appendChild(o); it.appendChild(ta);
  });
}

// NEW: add empty text box
function addBox(){
  if(!layout) return;
  const fs = parseInt(document.getElementById('fsize').value,10) || 24;
  const col = document.getElementById('fcol').value || '#111111';
  const w = Math.max(160, fs*6);
  const h = Math.max(60,  fs*2);
  const left = Math.round((img.naturalWidth  - w)/2);
  const top  = Math.round((img.naturalHeight - h)/2);

  // model
  layout.boxes = layout.boxes || [];
  layout.styles = layout.styles || [];
  layout.translations = layout.translations || [];
  layout.boxes.push({text:'', bbox:[left, top, w, h]});
  layout.styles.push({fontSize:fs, color:col});
  layout.translations.push('');

  // rebuild overlays & list to attach handlers properly
  buildOverlays();
  buildList();
  setActiveBox(layout.boxes.length-1);
  pushState();
}

function snapshot(){
  const brushPNG = brush.toDataURL('image/png');
  return {
    tr: [...document.querySelectorAll('#list textarea')].map(t=>t.value),
    bx: collectBoxes(),
    st: layout.styles ? JSON.parse(JSON.stringify(layout.styles)) : [],
    ec: eraserCount,
    rc: restoreCount,
    br: brushPNG
  };
}
async function applySnapshot(s){
  const ta = [...document.querySelectorAll('#list textarea')];
  s.tr.forEach((v,i)=>{ if(ta[i]) ta[i].value=v; const el=boxes[i]?.querySelector('.txt'); if(el) el.textContent=v; });
  s.bx.forEach((b,i)=>{ const box=boxes[i]; if(!box) return; const [x,y,w,h]=b.bbox; box.style.left=x+'px'; box.style.top=y+'px'; box.style.width=w+'px'; box.style.height=h+'px'; });
  layout.styles = s.st;
  await new Promise(res=>{ const im=new Image(); im.onload=()=>{ clearMask(); bctx.drawImage(im,0,0); res(); }; im.src=s.br; });

  if(lp){
    const r1=await fetch('/api/set_eraser_count',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({layout_path:lp,count:s.ec})});
    if(r1.ok){ const j=await r1.json(); eraserCount=j.eraser_count ?? s.ec; clean=j.clean_image || clean; }
    const r2=await fetch('/api/set_restore_count',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({layout_path:lp,count:s.rc})});
    if(r2.ok){ const j2=await r2.json(); restoreCount=j2.restore_count ?? s.rc; clean=j2.clean_image || clean; }
    await refreshImage(false);
  }
}
function pushState(){ history.push(snapshot()); if(history.length>100) history.shift(); future.length=0; }
function clearHistory(){ history.length=0; future.length=0; }
function undo(){ if(history.length<=1) return; const cur=history.pop(); future.push(cur); applySnapshot(history[history.length-1]); }
function redo(){ if(future.length===0) return; const s=future.pop(); history.push(s); applySnapshot(s); }

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
  autoFit(i);
  const r=await fetch('/api/update_style',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({layout_path:lp,index:i,fontSize:fs,color:col})});
  const j=await r.json(); if(!r.ok){ stat(j.error,true); return; }
  pushState(); stat('Style updated');
}
async function renderAndDownload(){
  if(!lp) return;
  await fetch('/api/update_boxes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({layout_path:lp, boxes:collectBoxes()})});
  await fetch('/api/render',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({layout_path:lp, translations:[...document.querySelectorAll('#list textarea')].map(t=>t.value)})});
  const items=encodeURIComponent(JSON.stringify(batch.length ? batch : [lp]));
  location.href='/api/download_zip?include=edited&items='+items;
  stat('Rendered & downloading ZIP…');
}
async function rebuild(){
  if(!lp) return;
  const r=await fetch('/api/rebuild_clean',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({layout_path:lp})});
  const j=await r.json(); if(!r.ok){ stat(j.error,true); return; }
  clean=j.clean_image; await refreshImage(false); stat('Clean updated');
}
function stat(m,err=false){ statusEl.innerHTML=err?('<span class="err">'+m+'</span>'):('<span class="ok">'+m+'</span>'); }

function startPaint(e){
  if(view!=='edited') return;
  painting=true;
  brush.style.pointerEvents='auto';

  bctx.globalCompositeOperation='source-over';
  bctx.globalAlpha = 1.0;

  bctx.lineWidth=+document.getElementById('bsize').value;
  bctx.lineCap='round'; bctx.lineJoin='round';
  const p=pt(e); bctx.beginPath(); bctx.moveTo(p.x,p.y);
}
function movePaint(e){
  if(!painting || view!=='edited') return;
  const p=pt(e); bctx.lineTo(p.x,p.y); bctx.stroke();
}
function endPaint(){ painting=false; }
function pt(e){ const r=brush.getBoundingClientRect(); return {x:(e.clientX-r.left)/scale, y:(e.clientY-r.top)/scale}; }
function clearMask(){ bctx.clearRect(0,0,brush.width,brush.height); }

function beginProgress(kind, strength){
  const t0 = performance.now();
  stat(`${kind}… (Strength: ${Math.round(strength*100)}%)`);
  const id = setInterval(()=>{
    const s = ((performance.now()-t0)/1000).toFixed(1);
    stat(`${kind}… (Strength: ${Math.round(strength*100)}%) ${s}s`);
  }, 120);
  return {t0, id};
}
function endProgress(timer, okText){
  clearInterval(timer.id);
  const s = ((performance.now()-timer.t0)/1000).toFixed(1);
  stat(`${okText} (${s}s)`);
}

async function applyMask(){
  if(!lp) return;
  const tmp=document.createElement('canvas'); tmp.width=brush.width; tmp.height=brush.height;
  const t=tmp.getContext('2d'); t.fillStyle='black'; t.fillRect(0,0,tmp.width,tmp.height);
  t.globalCompositeOperation='source-over'; t.drawImage(brush,0,0);
  const dataURL = tmp.toDataURL('image/png');
  clearMask();
  const strength = (+document.getElementById('strength').value || 100)/100;

  const timer = beginProgress('Erasing', strength);
  const r=await fetch('/api/erase_mask',{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({layout_path:lp,mask_data_url:dataURL,strength})
  });
  const j=await r.json();
  if(!r.ok){ endProgress(timer, 'Erase failed'); stat(j.error,true); return; }
  eraserCount = j.eraser_count ?? eraserCount;
  clean=j.clean_image; await refreshImage(false);
  endProgress(timer, 'Inpaint OK');
  pushState();
}

async function applyRestore(){
  if(!lp) return;
  const tmp=document.createElement('canvas'); tmp.width=brush.width; tmp.height=brush.height;
  const t=tmp.getContext('2d'); t.fillStyle='black'; t.fillRect(0,0,tmp.width,tmp.height);
  t.globalCompositeOperation='source-over'; t.drawImage(brush,0,0);
  const dataURL = tmp.toDataURL('image/png');
  clearMask();
  const strength = (+document.getElementById('strength').value || 100)/100;

  const timer = beginProgress('Restoring', strength);
  const r=await fetch('/api/restore_mask',{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({layout_path:lp,mask_data_url:dataURL,strength})
  });
  const j=await r.json();
  if(!r.ok){ endProgress(timer, 'Restore failed'); stat(j.error,true); return; }
  restoreCount = j.restore_count ?? restoreCount;
  clean=j.clean_image; await refreshImage(false);
  endProgress(timer, 'Restore OK');
  pushState();
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
