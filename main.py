# main.py
import argparse, json, os, pathlib, traceback, hashlib
from typing import List, Tuple
from tqdm import tqdm
from ocr_torii import ocr_image
from translate_lingva import translate_lines, translate_lines_impl
from renderer import render_translations
from utils import OCRBox
from config import DEFAULT_FONT_PATH
from PIL import Image

# ---------------- helpers ----------------

def _debug_dump(name, items, limit=5):
    try:
        print(f"[debug] {name} ({min(len(items), limit)} of {len(items)}):",
              [items[i] for i in range(min(len(items), limit))])
    except Exception:
        pass

def safe_ocr_image(path: str, api_key: str, out_dir: str, scales=(1.0, 0.5, 0.33, 0.25)):
    """OCR-г эхлээд жижгэрүүлэлгүй оролдож, бүтэхгүй бол хэд хэдэн жижгэрүүлэлтээр fallback хийх."""
    try:
        data = ocr_image(path, api_key)
        return data, 1.0
    except Exception:
        from PIL import Image as PILImage
        base = pathlib.Path(path)
        with PILImage.open(path) as im:
            w, h = im.size
            for s in scales:
                if s >= 1.0:
                    continue
                nw = max(1, int(w * s)); nh = max(1, int(h * s))
                scaled_path = None
                try:
                    scaled = im.resize((nw, nh), PILImage.LANCZOS)
                    scaled_path = pathlib.Path(out_dir) / (base.stem + f"_scaled_{int(s*100)}" + base.suffix)
                    scaled.save(scaled_path)
                    data = ocr_image(str(scaled_path), api_key)
                    return data, s
                except Exception:
                    if scaled_path:
                        try:
                            pathlib.Path(scaled_path).unlink()
                        except Exception:
                            pass
        raise

def polygon_to_bbox(polygon):
    xs = polygon[0::2]; ys = polygon[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

def _rebalance_to_boxes(boxes: List[OCRBox], sentences: List[str]) -> List[str]:
    """Нэгтгэсэн өгүүлбэрүүдийг талбарын талбайд пропорциональ байдлаар дахин хуваарилах."""
    if not boxes:
        return []
    full_text = " ".join([s for s in sentences if s and s.strip()]).strip()
    if not full_text:
        return [""] * len(boxes)
    words = full_text.split()
    areas = [(max(1, b.bbox[2]) * max(1, b.bbox[3])) for b in boxes]
    total_area = sum(areas) or len(boxes)
    targets = [max(1, int(round(len(words) * (a / total_area)))) for a in areas]
    diff = len(words) - sum(targets); i = 0
    while diff != 0 and len(targets) > 0:
        idx = i % len(targets)
        if diff > 0:
            targets[idx] += 1; diff -= 1
        else:
            if targets[idx] > 1:
                targets[idx] -= 1; diff += 1
        i += 1
    chunks=[]; widx=0
    for t in targets:
        chunks.append(" ".join(words[widx:widx+t]).strip()); widx += t
    if len(chunks) < len(boxes):
        chunks += [""] * (len(boxes)-len(chunks))
    elif len(chunks) > len(boxes):
        chunks = chunks[:len(boxes)]
    return chunks

# --------------- core pipeline ---------------

def _extract_boxes_from_ocr(ocr_json, scale_used: float) -> List[OCRBox]:
    """
    Torii OCR-н гаралтыг уян хатан парс хийж OCRBox[] болгоно.
    Дэмжлэг:
      - item['text'] + item['polygon'] (x1,y1,...,x4,y4)
      - item['text'] + item['bbox']    (x,y,w,h)
    """
    boxes: List[OCRBox] = []
    if not ocr_json:
        return boxes

    # Torii ихэвчлэн { "data": [ { "text": "...", "polygon": [x1,y1,...] }, ... ] } маягтай
    items = None
    if isinstance(ocr_json, dict) and 'data' in ocr_json:
        items = ocr_json.get('data') or []
    elif isinstance(ocr_json, list):
        items = ocr_json
    else:
        items = []

    for it in items:
        try:
            txt = (it.get('text') or "").strip()
            if not txt:
                continue

            if 'polygon' in it and it['polygon']:
                bbox = polygon_to_bbox(it['polygon'])
            elif 'bbox' in it and it['bbox']:
                bb = it['bbox']
                # bb may be {x:...,y:...,w:...,h:...} or [x,y,w,h]
                if isinstance(bb, dict):
                    bbox = (int(bb.get('x',0)), int(bb.get('y',0)),
                            int(bb.get('w',0)), int(bb.get('h',0)))
                elif isinstance(bb, (list, tuple)) and len(bb) >= 4:
                    bbox = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
                else:
                    continue
            else:
                continue

            # масштабаас буцааж нормчлох шаардлагагүй — OCR аль зураг дээр хийгдсэнтэй адил хэрэглэж буй.
            boxes.append(OCRBox(text=txt, bbox=bbox))
        except Exception:
            continue

    return boxes

def process_image(path: str, out_dir: str, font_path: str, max_chars: int, line_width: int,
                  api_key: str, manga_mode: bool = False, local_only: bool = False):

    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- эх зураг нээж шалгах ---
    img = Image.open(path).convert("RGB")
    w, h = img.size
    print(f"[debug] open {path} size={w}x{h}")
    if w < 50 or h < 50:
        img.close()
        raise ValueError("Image must be greater than 50x50 pixels")

    # --- OCR ---
    ocr_json, scale_used = safe_ocr_image(path, api_key, out_dir=out_dir)
    boxes: List[OCRBox] = _extract_boxes_from_ocr(ocr_json, scale_used)

    # зарим цэвэрлэгээ: хоосон/цагаан зай ихтэй текстүүдийг шүүх
    boxes = [b for b in boxes if b.text and b.text.strip()]
    _debug_dump("ocr_boxes", [b.bbox for b in boxes], limit=5)

    # OCR юу ч олдохгүй бол шууд эх зургийг копи хийгээд зөвхөн layout.json үүсгэнэ
    if not boxes:
        p = pathlib.Path(path)
        with open(p, "rb") as _f:
            uid8 = hashlib.sha1(_f.read()).hexdigest()[:8]
        out_base = pathlib.Path(out_dir) / f"{p.stem}-{uid8}"
        out_base.mkdir(parents=True, exist_ok=True)
        out_img  = out_base / p.name
        out_json = out_base / (p.stem + ".layout.json")

        img.save(out_img)
        layout = {
            "source_image": str(p.resolve()),
            "boxes": [],
            "translations": [],
            "note": "No OCR boxes detected; saved original image."
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(layout, f, ensure_ascii=False, indent=2)
        img.close()
        return str(out_img), str(out_json)

    # --- Орчуулга ---
    originals = [b.text for b in boxes]
    # Заримдаа manga balloon бүх текстийг нэг урт өгүүлбэр болгож орчуулбал дээр байдаг:
    # manga_mode=True үед бүх текстийг нэгтгээд буцаад талбайгаар rebalance хийе.
    if manga_mode:
        merged = [" ".join(originals)]
        try:
            translated = translate_lines(merged, max_chars=max_chars)
        except Exception:
            # хувилбар функц руу fallback
            translated = translate_lines_impl(merged)
        # дахин хуваарилалт
        translations = _rebalance_to_boxes(boxes, translated)
    else:
        try:
            translations = translate_lines(originals, max_chars=max_chars)
        except Exception:
            translations = translate_lines_impl(originals)

        # тоо нь зөрвөл тааруулж өгнө
        if len(translations) != len(boxes):
            translations = _rebalance_to_boxes(boxes, translations)

    _debug_dump("translations", translations, limit=5)

    # --- Рэндэр (inpaint/overlay) ---
    rendered_img = None
    try:
        # render_translations сигнатур янз бүр байх эрсдэлтэй тул эхлээд keyword-оор оролдож үзнэ
        try:
            rendered_img = render_translations(
                img=img,
                boxes=boxes,
                translations=translations,
                font_path=font_path,
                line_width=line_width,
                manga_mode=manga_mode,
                local_only=local_only
            )
        except TypeError:
            # Хэрэв keyword таарахгүй бол positional-оор дахин оролдъё:
            rendered_img = render_translations(img, boxes, translations, font_path, line_width)
    except Exception as e:
        print(f"[warn] render_translations failed: {e}")
        # fallback: рэндэр хийхгүй, original зургийг бичнэ
        rendered_img = img

    # --- Гаралтын файлууд ---
    p = pathlib.Path(path)
    with open(p, "rb") as _f:
        uid8 = hashlib.sha1(_f.read()).hexdigest()[:8]
    out_base = pathlib.Path(out_dir) / f"{p.stem}-{uid8}"
    out_base.mkdir(parents=True, exist_ok=True)

    out_img  = out_base / p.name
    out_json = out_base / (p.stem + ".layout.json")

    # PIL.Image эсэхийг шалгаад хадгална
    if isinstance(rendered_img, Image.Image):
        rendered_img.save(out_img)
        # rendered_img нь img-тай ижил объект байж магадгүй — давхар хаахгүйгээр аюулгүй байхын тулд copy хийж ашигласан гэж үзье
    else:
        # хэрэв renderer өөр төрлийн буцаалт өгвөл original-г хадгал
        img.save(out_img)

    layout = {
        "source_image": str(p.resolve()),
        "boxes": [{"text": b.text, "bbox": b.bbox} for b in boxes],
        "translations": translations,
        "scale_used": scale_used
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(layout, f, ensure_ascii=False, indent=2)

    # эх зургийн handle-ийг хаана
    try:
        img.close()
    except Exception:
        pass

    return str(out_img), str(out_json)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="input")
    ap.add_argument("--out", dest="out_dir", default="output")
    ap.add_argument("--font", dest="font_path", default=DEFAULT_FONT_PATH)
    ap.add_argument("--max-chars", type=int, default=800)
    ap.add_argument("--line-width", type=int, default=24)
    ap.add_argument("--api-key", dest="api_key", default=os.getenv("TORII_API_KEY"))
    ap.add_argument("--manga-mode", dest="manga_mode", action="store_true")
    ap.add_argument("--local-only", dest="local_only", action="store_true")
    args = ap.parse_args()

    images: List[pathlib.Path] = []
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        images += list(pathlib.Path(args.in_dir).glob(f"*{ext}"))
    if not images:
        print(f"[WARN] No images found in: {args.in_dir}")
        return

    for img_path in tqdm(images, desc="Translating"):
        try:
            out_img, out_json = process_image(
                str(img_path), args.out_dir, args.font_path,
                args.max_chars, args.line_width,
                args.api_key, manga_mode=args.manga_mode,
                local_only=args.local_only
            )
            tqdm.write(f"[OK] {img_path.name} -> {out_img}")
        except Exception as ex:
            tb = traceback.format_exc()
            tqdm.write(f"[FAIL] {img_path.name}: {ex}\n{tb}")

if __name__ == "__main__":
    main()
