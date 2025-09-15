# main.py (clean, fixed)
import argparse, json, os, pathlib, traceback
from typing import List
from tqdm import tqdm
from ocr_torii import ocr_image
from style import is_sfx
from translate_lingva import translate_lines, translate_lines_impl   # ✅ Lingvanex
from renderer import render_translations
from utils import OCRBox
from config import DEFAULT_FONT_PATH
from PIL import Image


def _debug_dump(name, items, limit=5):
    try:
        print(f"[debug] {name} ({min(len(items), limit)} of {len(items)}):",
              [items[i] for i in range(min(len(items), limit))])
    except Exception:
        pass


def safe_ocr_image(path: str, api_key: str, out_dir: str, scales=(1.0, 0.5, 0.33, 0.25)):
    """Call OCR and, on failure, try smaller scaled versions saved to out_dir.
    Returns (ocr_dict, scale_used). If no scaling needed returns scale 1.0.
    """
    try:
        data = ocr_image(path, api_key)
        return data, 1.0
    except Exception:
        # try downscales
        from PIL import Image as PILImage
        base = pathlib.Path(path)
        with PILImage.open(path) as im:
            w, h = im.size
            for s in scales:
                if s >= 1.0:
                    continue
                nw = max(1, int(w * s))
                nh = max(1, int(h * s))
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
        # re-raise if none worked
        raise


def polygon_to_bbox(polygon):
    # polygon: [x1,y1,x2,y2,x3,y3,x4,y4]
    xs = polygon[0::2]
    ys = polygon[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def process_image(path: str, out_dir: str, font_path: str, max_chars: int, line_width: int,
                  api_key: str, manga_mode: bool = False, local_only: bool = False):
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ---------- Open & validate ----------
    with Image.open(path) as img_check:
        w, h = img_check.size
        print(f"[debug] open {path} size={w}x{h}")
        if w < 50 or h < 50:
            raise ValueError("Image must be greater than 50x50 pixels")

        max_dim = 10000
        max_parts_limit = 6
        required_parts = (h + max_dim - 1) // max_dim

        temp_paths: List[tuple[str, int]] = []  # (tile_path, top_offset)
        path_to_process = path
        processed_early = False  # if True, we already produced (img, boxes, translations)

        # ---------- Large image handling ----------
        if w > max_dim or h > max_dim:
            # crop width if needed
            if w > max_dim:
                img_check = img_check.crop((0, 0, max_dim, h))
                w = max_dim

            if required_parts > max_parts_limit:
                # Downscale whole image first, OCR on scaled, map boxes back
                scale = (max_parts_limit * max_dim) / float(h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                scaled = img_check.resize((new_w, new_h), resample=Image.LANCZOS)
                scaled_path = pathlib.Path(out_dir) / (pathlib.Path(path).stem + "_scaled" + pathlib.Path(path).suffix)
                scaled.save(scaled_path)
                print(f"[debug] downscaling image to {new_w}x{new_h} (scale={scale:.4f}) -> {scaled_path}")

                # OCR on scaled; map to original coords
                ocr, used_scale = safe_ocr_image(str(scaled_path), api_key, out_dir)
                boxes: List[OCRBox] = []
                for l in ocr.get("lines", []):
                    if l.get("text") and l.get("boundingBox") and len(l["boundingBox"]) == 8:
                        sx, sy, sw, sh = polygon_to_bbox(l["boundingBox"])
                        if used_scale and used_scale != 1.0:
                            ox = int(round(sx / used_scale))
                            oy = int(round(sy / used_scale))
                            ow = int(round(sw / used_scale))
                            oh = int(round(sh / used_scale))
                        else:
                            ox, oy, ow, oh = sx, sy, sw, sh
                        boxes.append(OCRBox(text=l["text"], bbox=(ox, oy, ow, oh)))

                src_lines = [b.text.strip()[:max_chars] for b in boxes]
                _debug_dump("src_lines", src_lines)
                if manga_mode:
                    manga_sys = ("You are a manga translator: produce concise, natural, balloon-friendly Mongolian (Cyrillic). "
                                 "Keep the tone casual and short; fit speech balloons.")
                    translations = translate_lines_impl(src_lines, system_override=manga_sys, local_only=local_only)
                else:
                    # ❌ reflow-г painter дээр хэрэглэхгүй
                    translations = translate_lines(src_lines) if not local_only else translate_lines_impl(src_lines, local_only=True)
                _debug_dump("translations", translations)

                img = render_translations(path, boxes, translations, font_path, line_width=line_width)

                # cleanup temp scale
                try:
                    pathlib.Path(scaled_path).unlink()
                except Exception:
                    pass

                processed_early = True
            else:
                # Tile vertically (<= max_dim) with small overlap
                overlap = 40
                y = 0
                part_idx = 0
                while y < h:
                    top = y
                    bottom = min(h, y + max_dim)
                    crop = img_check.crop((0, top, w, bottom))
                    temp_path = pathlib.Path(out_dir) / f"{pathlib.Path(path).stem}_part{part_idx}{pathlib.Path(path).suffix}"
                    crop.save(temp_path)

                    # sanity check tile size
                    from PIL import Image as PILImage
                    with PILImage.open(temp_path) as timg:
                        tw, th = timg.size
                    if tw > max_dim or th > max_dim:
                        print(f"[debug] tile {temp_path} size={tw}x{th} exceeds {max_dim}, splitting...")
                        try:
                            pathlib.Path(temp_path).unlink()
                        except Exception:
                            pass
                        subparts = (th + max_dim - 1) // max_dim
                        for si in range(subparts):
                            sub_top = si * max_dim
                            sub_bottom = min(th, sub_top + max_dim)
                            with PILImage.open(path) as _orig:
                                sub_crop = _orig.crop((0, top + sub_top, w, top + sub_bottom))
                            sub_temp = pathlib.Path(out_dir) / f"{pathlib.Path(path).stem}_part{part_idx}{pathlib.Path(path).suffix}"
                            sub_crop.save(sub_temp)
                            temp_paths.append((str(sub_temp), top + sub_top))
                            part_idx += 1
                        if bottom >= h:
                            break
                        y = bottom - overlap
                        continue

                    temp_paths.append((str(temp_path), top))
                    part_idx += 1
                    if bottom >= h:
                        break
                    y = bottom - overlap

    # ---------- If not processed early, do normal flow ----------
    if not processed_early:
        all_boxes: List[OCRBox] = []
        translations: List[str] = []

        if temp_paths:
            # Process tiles then stitch
            from PIL import Image as PILImage
            rendered_parts = []
            print(f"[debug] will process {len(temp_paths)} parts: {[p for p,_ in temp_paths]}")
            for part_path, top_offset in temp_paths:
                ocr, used_scale = safe_ocr_image(part_path, api_key, out_dir)
                boxes: List[OCRBox] = []
                for l in ocr.get("lines", []):
                    if l.get("text") and l.get("boundingBox") and len(l["boundingBox"]) == 8:
                        bx, by, bw, bh = polygon_to_bbox(l["boundingBox"])
                        if used_scale and used_scale != 1.0:
                            bx = int(round(bx / used_scale))
                            by = int(round(by / used_scale))
                            bw = int(round(bw / used_scale))
                            bh = int(round(bh / used_scale))
                        boxes.append(OCRBox(text=l["text"], bbox=(bx, by, bw, bh)))

                src_lines = [b.text.strip()[:max_chars] for b in boxes]
                _debug_dump("src_lines(part)", src_lines)
                if manga_mode:
                    manga_sys = ("You are a manga translator: produce concise, natural, balloon-friendly Mongolian (Cyrillic). "
                                 "Keep the tone casual and short; fit speech balloons.")
                    part_translations = translate_lines_impl(src_lines, system_override=manga_sys, local_only=local_only)
                else:
                    # ❌ reflow-г painter дээр хэрэглэхгүй
                    part_translations = translate_lines(src_lines) if not local_only else translate_lines_impl(src_lines, local_only=True)
                _debug_dump("translations(part)", part_translations)

                rendered = render_translations(part_path, boxes, part_translations, font_path, line_width=line_width)
                rendered_parts.append((rendered, top_offset))

                for b, t in zip(boxes, part_translations):
                    bx, by, bw, bh = b.bbox
                    all_boxes.append(OCRBox(text=b.text, bbox=(bx, by + top_offset, bw, bh)))
                    translations.append(t)

            # stitch
            with Image.open(path) as im0:
                w, h = im0.size
            full_img = PILImage.new("RGBA", (w, h), (255, 255, 255, 0))
            for rendered, top in rendered_parts:
                full_img.paste(rendered, (0, top))
            img = full_img
            boxes = all_boxes

        else:
            # Single pass
            print(f"[debug] processing single image at: {path}")
            ocr, used_scale = safe_ocr_image(path, api_key, out_dir)
            boxes: List[OCRBox] = []
            for l in ocr.get("lines", []):
                if l.get("text") and l.get("boundingBox") and len(l["boundingBox"]) == 8:
                    bx, by, bw, bh = polygon_to_bbox(l["boundingBox"])
                    if used_scale and used_scale != 1.0:
                        bx = int(round(bx / used_scale))
                        by = int(round(by / used_scale))
                        bw = int(round(bw / used_scale))
                        bh = int(round(bh / used_scale))
                    boxes.append(OCRBox(text=l["text"], bbox=(bx, by, bw, bh)))

            src_lines = [b.text.strip()[:max_chars] for b in boxes]
            _debug_dump("src_lines", src_lines)
            if manga_mode:
                manga_sys = ("You are a manga translator: produce concise, natural, balloon-friendly Mongolian (Cyrillic). "
                             "Keep the tone casual and short; fit speech balloons.")
                translations = translate_lines_impl(src_lines, system_override=manga_sys, local_only=local_only)
            else:
                # ❌ reflow-г painter дээр хэрэглэхгүй
                translations = translate_lines(src_lines) if not local_only else translate_lines_impl(src_lines, local_only=True)
            _debug_dump("translations", translations)

            img = render_translations(path, boxes, translations, font_path, line_width=line_width)

    # ---------- Save outputs ----------
    p = pathlib.Path(path)
    out_img = pathlib.Path(out_dir) / p.name
    out_json = pathlib.Path(out_dir) / (p.stem + ".layout.json")
    out_img.parent.mkdir(parents=True, exist_ok=True)

    img.save(out_img)
    layout = {
        "source_image": str(p),
        "boxes": [{"text": b.text, "bbox": b.bbox} for b in boxes],
        "translations": translations,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(layout, f, ensure_ascii=False, indent=2)

    # cleanup temp tiles
    for tp, _ in (temp_paths or []):
        try:
            pathlib.Path(tp).unlink()
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
    ap.add_argument("--manga-mode", dest="manga_mode", action="store_true",
                    help="Use manga-optimized translation prompt (short, balloon-friendly Mongolian)")
    ap.add_argument("--local-only", dest="local_only", action="store_true",
                    help="Run without external calls; use glossary/local rules for translations")
    args = ap.parse_args()

    images: List[pathlib.Path] = []
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        images += list(pathlib.Path(args.in_dir).glob(f"*{ext}"))
    if not images:
        print(f"[WARN] No images found in: {args.in_dir}")
        return

    for img in tqdm(images, desc="Translating"):
        try:
            out_img, out_json = process_image(
                str(img), args.out_dir, args.font_path,
                args.max_chars, args.line_width,
                args.api_key, manga_mode=args.manga_mode,
                local_only=args.local_only
            )
            tqdm.write(f"[OK] {img.name} -> {out_img}")
        except Exception as ex:
            tb = traceback.format_exc()
            tqdm.write(f"[FAIL] {img.name}: {ex}\n{tb}")


if __name__ == "__main__":
    main()
