from typing import List
from PIL import Image, ImageDraw, ImageFont
from utils import OCRBox, estimate_bg_color, draw_soft_rect, fit_font
from style import is_sfx
from glossary import KEEP_AS_IS

BOX_PADDING = 6

def render_translations(img_path: str, boxes: List[OCRBox], translations: List[str], font_path: str, line_width: int = 22):
    # Хамгаалалт – pipeline асуудлыг тодорхой мессежтэй унага
    if translations is None:
        raise ValueError("render_translations: translations is None (upstream translate_lines_impl буцаагаагүй)")
    if len(boxes) != len(translations):
        raise ValueError(f"render_translations: count mismatch boxes={len(boxes)} vs trans={len(translations)}")

    img = Image.open(img_path).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

        # --- guard: align translations length with boxes length ---
    if translations is None:
        # no translations? -> keep originals so we skip drawing (src==dst)
        translations = [b.text for b in boxes]
    elif len(translations) != len(boxes):
        # normalize lengths: truncate or pad with original text
        if len(translations) > len(boxes):
            translations = translations[:len(boxes)]
        else:
            # pad with originals so those indices will be skipped (src == dst)
            translations = translations + [b.text for b in boxes[len(translations):]]


    # First pass: sample local background color per box and draw a soft rect to match
    box_rects = []
    box_text_colors = []
    for box in boxes:
        x, y, w, h = box.bbox
        x -= BOX_PADDING; y -= BOX_PADDING; w += BOX_PADDING*2; h += BOX_PADDING*2
        box_rects.append((x, y, w, h))
        # estimate background color around the box
        try:
            bg = estimate_bg_color(img, (x, y, w, h))
        except Exception:
            bg = (255, 255, 255)
        # ensure RGBA tuple
        if isinstance(bg, tuple) and len(bg) == 3:
            rect_color = (bg[0], bg[1], bg[2], 255)
        elif isinstance(bg, tuple) and len(bg) == 4:
            rect_color = bg
        else:
            rect_color = (255, 255, 255, 255)

        # Fill the box with sampled background color
        try:
            draw_soft_rect(draw, img, (x, y, w, h), rect_color, blur_radius=0, corner=6)
        except Exception:
            draw.rectangle([x, y, x + w, y + h], fill=rect_color)

        # choose contrasting text color (black or white)
        r, g, b = rect_color[0], rect_color[1], rect_color[2]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        if luminance > 150:
            text_fill = (0, 0, 0, 255)
        else:
            text_fill = (255, 255, 255, 255)

        box_text_colors.append((text_fill, None))

    # Render
    for idx, (box, text) in enumerate(zip(box_rects, translations)):
        if not text or not text.strip():
            continue
        # SFX эсвэл KEEP_AS_IS эсвэл орчуулга өөрчлөгдөөгүй бол зураг дээр дахин бичихгүй
        src_text = boxes[idx].text.strip()
        if is_sfx(src_text) or src_text.upper() in (KEEP_AS_IS or set()) or src_text.strip() == text.strip():
            continue

        x, y, w, h = box
        orig_box = boxes[idx]
        x0, y0, w0, h0 = orig_box.bbox
        x0 -= BOX_PADDING; y0 -= BOX_PADDING; w0 += BOX_PADDING*2; h0 += BOX_PADDING*2

        est_size = max(12, int(h0 * 0.55))
        est_size = min(est_size, 128)

        try:
            font_try, wrapped = fit_font(font_path, text, w0, h0, max_size=est_size, min_size=est_size, line_width=line_width)
            font = font_try
        except Exception:
            font, wrapped = fit_font(font_path, text, w0, h0, line_width=line_width)

        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=2, align="center")
        tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        size = getattr(font, "size", None)
        min_size = 10
        while (tw > w or th > h) and size and size > min_size:
            size = max(min_size, size - 2)
            try:
                font = ImageFont.truetype(font_path, int(size))
            except Exception:
                break
            font2, wrapped = fit_font(font_path, text, w, h, max_size=int(size), min_size=int(min_size), line_width=line_width)
            font = font2
            bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=2, align="center")
            tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]

        if tw > w or th > h:
            lines = wrapped.splitlines()
            new_lines = []
            for ln in lines:
                while ln and draw.multiline_textbbox((0,0), "\n".join(new_lines + [ln + "…"]), font=font, spacing=2)[3] > h:
                    ln = ln[:-1]
                if not ln:
                    continue
                if ln != lines[-1] and not ln.endswith("…"):
                    new_lines.append(ln)
                else:
                    new_lines.append(ln + ("…" if not ln.endswith("…") else ""))
            wrapped = "\n".join(new_lines).strip()
            bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=2, align="center")
            tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]

        tx = x + (w - tw) / 2
        ty = y + (h - th) / 2

        try:
            tex_fill, _ = box_text_colors[idx]
        except Exception:
            tex_fill = (0,0,0,255)
        stroke_fill = (255, 255, 255, 255) if tex_fill[0] == 0 else (0,0,0,255)

        try:
            draw.multiline_text((tx, ty), wrapped, font=font, fill=tex_fill, align="center", spacing=2, stroke_width=1, stroke_fill=stroke_fill)
        except TypeError:
            draw.multiline_text((tx+1, ty+1), wrapped, font=font, fill=(255,255,255,255), align="center", spacing=2)
            draw.multiline_text((tx, ty), wrapped, font=font, fill=tex_fill, align="center", spacing=2)

    return img
