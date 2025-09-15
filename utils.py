from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageStat
import math

@dataclass
class OCRBox:
    text: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)

def estimate_bg_color(img, box, expand=4):
    """Estimate a robust background RGB color for the given box.

    Strategy:
    - Expand the box by a padding region and sample pixels from the surrounding area.
    - Ignore pixels that fall inside the central text area (optionally with a small inner pad).
    - Quantize peripheral pixels into coarse buckets and pick the most frequent bucket as
      the background color, then average full-resolution colors from that bucket.
    - If sampling fails or is insufficient, fall back to the mean color of the crop.
    """
    x, y, w, h = box
    pad = max(2, min(w, h) // 6) * expand
    W, H = img.size
    sx = max(0, x - pad); sy = max(0, y - pad)
    ex = min(W, x + w + pad); ey = min(H, y + h + pad)
    crop = img.crop((sx, sy, ex, ey)).convert("RGB")

    cw, ch = crop.size
    # inner box coordinates relative to crop
    ix = max(0, x - sx);
    iy = max(0, y - sy);
    iw = min(w, cw - ix);
    ih = min(h, ch - iy)

    # inner pad to avoid bleed from text area itself
    inner_pad = max(2, min(iw, ih) // 8)

    # --- New: check inner area for a uniform/light bubble color ---
    try:
        inner_box = crop.crop((max(0, ix - inner_pad), max(0, iy - inner_pad), min(cw, ix + iw + inner_pad), min(ch, iy + ih + inner_pad)))
        stat_inner = ImageStat.Stat(inner_box)
        # mean luminance of inner area
        r_mean, g_mean, b_mean = stat_inner.mean[:3]
        lum_inner = 0.299 * r_mean + 0.587 * g_mean + 0.114 * b_mean
        # RMS/variance check to detect if inner area is fairly uniform (bubble) vs noisy/textured
        rms_vals = stat_inner.rms[:3]
        rms_mean = sum(rms_vals) / 3.0
        # If inner area is light and has low variance, prefer it as the bubble color
        if lum_inner > 120 and rms_mean < 30:
            return (int(round(r_mean)), int(round(g_mean)), int(round(b_mean)))
    except Exception:
        # silently ignore and continue to peripheral sampling
        pass

    # sample peripheral pixels with a stride to limit work on large crops
    stride = max(1, min(cw, ch) // 160)
    pixels = crop.load()
    buckets = {}
    bucket_members = {}
    total_samples = 0
    for py in range(0, ch, stride):
        for px in range(0, cw, stride):
            # skip pixels that are inside the inner text area (with inner_pad)
            if (ix - inner_pad) <= px < (ix + iw + inner_pad) and (iy - inner_pad) <= py < (iy + ih + inner_pad):
                continue
            r, g, b = pixels[px, py]
            # exclude likely ink/text pixels: very dark or near-grayscale high-contrast
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            mx = max(r, g, b); mn = min(r, g, b)
            sat = 0 if mx == 0 else (mx - mn) / mx
            # skip very dark pixels or low-saturation pixels that are likely text/line-art
            if lum < 40 or (sat < 0.12 and lum < 200):
                continue
            # quantize to 16 levels per channel to bucket similar colors
            key = ((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4)
            buckets[key] = buckets.get(key, 0) + 1
            bucket_members.setdefault(key, []).append((r, g, b))
            total_samples += 1

    # If no peripheral samples (very small crop), fall back to full-crop mean
    if total_samples == 0:
        stat = ImageStat.Stat(crop)
        mean = tuple(int(v) for v in stat.mean[:3])
        return mean

    # pick the most frequent bucket
    best_key = max(buckets.items(), key=lambda kv: kv[1])[0]
    members = bucket_members.get(best_key, [])

    # if the chosen bucket has too few members, fallback to median of all peripheral samples
    if len(members) < max(4, total_samples // 20):
        # collect all peripheral samples flattened
        all_samples = []
        for lst in bucket_members.values():
            all_samples.extend(lst)
        if not all_samples:
            stat = ImageStat.Stat(crop)
            mean = tuple(int(v) for v in stat.mean[:3])
            return mean
        # median per channel
        rs = sorted([p[0] for p in all_samples])
        gs = sorted([p[1] for p in all_samples])
        bs = sorted([p[2] for p in all_samples])
        mid = len(all_samples) // 2
        med = (rs[mid], gs[mid], bs[mid])
        return tuple(int(c) for c in med)

    # average full-resolution colors in best bucket
    ar = sum(p[0] for p in members) / len(members)
    ag = sum(p[1] for p in members) / len(members)
    ab = sum(p[2] for p in members) / len(members)
    return (int(round(ar)), int(round(ag)), int(round(ab)))

def draw_soft_rect(draw, img, box, color, blur_radius=2, corner=6):
    # Paint solid, then blur edges by compositing a blurred mask
    x, y, w, h = box
    rect = Image.new("RGBA", img.size, (0,0,0,0))
    rdraw = ImageDraw.Draw(rect)
    # color format fix
    if isinstance(color, tuple):
        if len(color) == 3:
            fill_color = color + (255,)
        elif len(color) == 4:
            fill_color = color
        else:
            raise ValueError("Color must be tuple of 3 (RGB) or 4 (RGBA) elements")
    else:
        fill_color = (255,255,255,255)
    rdraw.rounded_rectangle([x, y, x+w, y+h], radius=corner, fill=fill_color)
    blurred = rect.filter(ImageFilter.GaussianBlur(blur_radius))
    img.alpha_composite(blurred)

def fit_font(font_path, text, box_w, box_h, max_size=44, min_size=14, line_width=22):
    import textwrap

    # create temp image/draw for accurate measurement
    tmp_img = Image.new("RGBA", (max(1, box_w), max(1, box_h)))
    tmp_draw = ImageDraw.Draw(tmp_img)

    # Helper: try to wrap by pixel width using textwrap by testing multiple line widths
    def wrap_for_font(fnt):
        # start with a conservative chars-per-line estimate, then refine
        # use fallback widths between 8 and 80 chars to find the best pixel fit
        best = None
        for chars in range(max(8, int(line_width * 0.6)), max(8, int(line_width * 2)) + 1, 2):
            candidate = textwrap.fill(text, width=chars)
            bbox = tmp_draw.multiline_textbbox((0, 0), candidate, font=fnt, spacing=2, align="center")
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            if tw <= box_w and th <= box_h:
                # prefer larger chars (less wrapping) so pick the candidate with largest chars that fits
                best = (candidate, tw, th, chars)
        if best is not None:
            return best[0], best[1], best[2]
        # if nothing fits, return the tightest (most-wrapped) candidate measured
        # choose the smallest tw that is > box_w or smallest th that is > box_h
        tightest = None
        for chars in range(max(8, int(line_width * 0.6)), max(8, int(line_width * 2)) + 1, 2):
            candidate = textwrap.fill(text, width=chars)
            bbox = tmp_draw.multiline_textbbox((0, 0), candidate, font=fnt, spacing=2, align="center")
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            score = (max(0, tw - box_w) + max(0, th - box_h))
            if tightest is None or score < tightest[0]:
                tightest = (score, candidate, tw, th)
        return tightest[1], tightest[2], tightest[3]

    # try sizes from big to small, but prefer sizes that leave some padding
    for size in range(max_size, min_size - 1, -1):
        try:
            font = ImageFont.truetype(font_path, size)
        except Exception:
            # if font fails for this size, skip
            continue
        wrapped, tw, th = wrap_for_font(font)
        # allow small margin so text isn't touching edges
        if tw <= box_w - 4 and th <= box_h - 4:
            return font, wrapped

    # fallback: pick smallest size and its best wrap
    font = ImageFont.truetype(font_path, min_size)
    wrapped, tw, th = wrap_for_font(font)
    return font, wrapped
