# utils.py
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageStat
import os, base64, io
import numpy as np

# -------- Optional deps --------
try:
    import cv2
    _HAVE_CV = True
except Exception:
    _HAVE_CV = False

try:
    import requests
    _HAVE_REQ = True
except Exception:
    _HAVE_REQ = False


@dataclass
class OCRBox:
    text: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)


# ================= BG estimation =================
def estimate_bg_color(img: Image.Image, box: Tuple[int, int, int, int], expand: int = 4) -> Tuple[int, int, int]:
    x, y, w, h = box
    pad = max(2, min(w, h) // 6) * expand
    W, H = img.size
    sx = max(0, x - pad); sy = max(0, y - pad)
    ex = min(W, x + w + pad); ey = min(H, y + h + pad)
    crop = img.crop((sx, sy, ex, ey)).convert("RGB")

    cw, ch = crop.size
    ix = max(0, x - sx); iy = max(0, y - sy)
    iw = min(w, cw - ix); ih = min(h, ch - iy)
    inner_pad = max(2, min(iw, ih) // 8)

    # 1) center flat area shortcut
    try:
        inner = crop.crop((max(0, ix - inner_pad), max(0, iy - inner_pad),
                           min(cw, ix + iw + inner_pad), min(ch, iy + ih + inner_pad)))
        st = ImageStat.Stat(inner)
        r, g, b = st.mean[:3]
        lum = 0.299*r + 0.587*g + 0.114*b
        rms = sum(st.rms[:3]) / 3.0
        if lum > 120 and rms < 30:
            return int(round(r)), int(round(g)), int(round(b))
    except Exception:
        pass

    # 2) peripheral sampling (bucket)
    stride = max(1, min(cw, ch) // 160)
    pix = crop.load()
    buckets: Dict[int, int] = {}
    members: Dict[int, List[Tuple[int, int, int]]] = {}
    total = 0
    for py in range(0, ch, stride):
        for px in range(0, cw, stride):
            if (ix - inner_pad) <= px < (ix + iw + inner_pad) and (iy - inner_pad) <= py < (iy + ih + inner_pad):
                continue
            r, g, b = pix[px, py]
            lum = 0.299*r + 0.587*g + 0.114*b
            mx, mn = max(r, g, b), min(r, g, b)
            sat = 0 if mx == 0 else (mx - mn) / mx
            if lum < 40 or (sat < 0.12 and lum < 200):
                continue
            key = ((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4)
            buckets[key] = buckets.get(key, 0) + 1
            members.setdefault(key, []).append((r, g, b))
            total += 1

    if total == 0:
        m = ImageStat.Stat(crop).mean[:3]
        return int(m[0]), int(m[1]), int(m[2])

    best = max(buckets.items(), key=lambda kv: kv[1])[0]
    mem = members.get(best, [])
    if len(mem) < max(4, total // 20):
        alls = [p for lst in members.values() for p in lst]
        if not alls:
            m = ImageStat.Stat(crop).mean[:3]
            return int(m[0]), int(m[1]), int(m[2])
        rs = sorted(p[0] for p in alls); gs = sorted(p[1] for p in alls); bs = sorted(p[2] for p in alls)
        mid = len(alls) // 2
        return int(rs[mid]), int(gs[mid]), int(bs[mid])

    ar = sum(p[0] for p in mem) / len(mem); ag = sum(p[1] for p in mem) / len(mem); ab = sum(p[2] for p in mem) / len(mem)
    return int(round(ar)), int(round(ag)), int(round(ab))


# ================= helpers =================
def _rect_mask(size: Tuple[int, int], box: Tuple[int, int, int, int], feather: int = 0, corner: int = 6) -> Image.Image:
    W, H = size
    x, y, w, h = box
    mask_l = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask_l)
    d.rounded_rectangle([x, y, x + w, y + h], radius=corner, fill=255)
    if feather > 0:
        mask_l = mask_l.filter(ImageFilter.GaussianBlur(feather))
    return mask_l

def _mask_coverage(mask_u8: np.ndarray, box: Tuple[int, int, int, int]) -> float:
    x, y, w, h = box
    H, W = mask_u8.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x+w), min(H, y+h)
    if x2 <= x1 or y2 <= y1: return 0.0
    crop = mask_u8[y1:y2, x1:x2]
    return float(np.count_nonzero(crop)) / float(max(1, crop.size))

def _grow_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or not _HAVE_CV:
        return mask
    k = max(3, int(px) | 1)
    return cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)

def _diff_score(a: Image.Image, b: Image.Image, box: Tuple[int, int, int, int]) -> float:
    x, y, w, h = box
    a = a.convert("L"); b = b.convert("L")
    arr_a = np.array(a)[y:y+h, x:x+w].astype(np.int16)
    arr_b = np.array(b)[y:y+h, x:x+w].astype(np.int16)
    if arr_a.size == 0 or arr_b.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr_a - arr_b)))

# ================= Mask builders =================
def _build_text_mask(img: Image.Image,
                     box: Tuple[int, int, int, int],
                     expand_px: int = 4,
                     grow_px: Optional[int] = None):
    if not _HAVE_CV:
        return None
    x, y, w, h = box
    x = max(0, x - expand_px); y = max(0, y - expand_px)
    w += expand_px*2; h += expand_px*2

    arr = np.array(img.convert("RGB"))
    H, W = arr.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return None

    roi = arr[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    lab  = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB); L = lab[..., 0]
    hsv  = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV); V = hsv[..., 2]

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    ad = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 7)
    _, o1 = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, o2 = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(g, 50, 120)

    m = cv2.max(ad, cv2.max(o1, o2))
    m = cv2.max(m, edges)

    mean_v = float(np.mean(V))
    close_it = 2 if mean_v < 110 else 1
    dilate_it = 2 if mean_v < 110 else 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=close_it)
    m = cv2.dilate(m, np.ones((3,3),np.uint8), iterations=dilate_it)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(m)
    for c in cnts:
        if cv2.contourArea(c) >= 8:
            cv2.drawContours(mask, [c], -1, 255, -1)

    if grow_px is None:
        grow_px = int(max(3, min(9, 0.035 * min(w, h))))
    mask = _grow_mask(mask, grow_px)
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    full = np.zeros((H, W), dtype=np.uint8)
    full[y1:y2, x1:x2] = mask
    return full

def _b64_png(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

_LAMA_FAILS = 0

# ================= Generic HTTP inpaint =================
def _inpaint_http(img: Image.Image, mask_u8: np.ndarray):
    import io as _io, re
    if not _HAVE_REQ or mask_u8 is None:
        return None

    global _LAMA_FAILS
    if _LAMA_FAILS >= 5:
        return None

    base = os.getenv("INPAINT_URL", "http://127.0.0.1:8090").rstrip("/")
    candidates: List[str] = []
    if base.endswith(("/inpaint", "/api/inpaint", "/api/v1/inpaint")):
        candidates.append(base)
    else:
        candidates.extend([base + "/api/v1/inpaint", base + "/api/inpaint", base + "/inpaint"])

    model = os.getenv("INPAINT_MODEL", "lama")
    limit = int(os.getenv("INPAINT_SIZE", "4096"))
    hd_strategy = os.getenv("INPAINT_HD", "Original")

    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    pad = 32
    W, H = img.size
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(W, int(xs.max()) + pad + 1)
    y2 = min(H, int(ys.max()) + pad + 1)

    crop_img = img.crop((x1, y1, x2, y2)).convert("RGB")
    crop_mask = Image.fromarray(mask_u8[y1:y2, x1:x2]).convert("L")

    cw, ch = crop_img.size
    if max(cw, ch) > limit:
        if cw >= ch:
            nw = limit; nh = int(ch * (limit / cw))
        else:
            nh = limit; nw = int(cw * (limit / ch))
        crop_img = crop_img.resize((nw, nh), Image.LANCZOS)
        crop_mask = crop_mask.resize((nw, nh), Image.NEAREST)

    def _png_bytes(pil_im: Image.Image) -> bytes:
        b = _io.BytesIO(); pil_im.save(b, format="PNG"); return b.getvalue()

    img_bytes = _png_bytes(crop_img)
    mask_bytes = _png_bytes(crop_mask)

    img_data_url = "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")
    mask_data_url = "data:image/png;base64," + base64.b64encode(mask_bytes).decode("utf-8")

    def _decode_json_image(s: str) -> Image.Image:
        s = re.sub(r"^data:image/[^;]+;base64,", "", s or "")
        return Image.open(_io.BytesIO(base64.b64decode(s))).convert("RGBA")

    payloads = [
        ("json", None, {"image": img_data_url, "mask": mask_data_url, "model": model, "size_limit": limit, "hd_strategy": hd_strategy}),
        ("form", None, {"image": img_data_url, "mask": mask_data_url, "model": model, "size_limit": str(limit), "hd_strategy": hd_strategy}),
        ("multipart", {"image": ("image.png", img_bytes, "image/png"), "mask": ("mask.png", mask_bytes, "image/png")}, {"model": model, "size_limit": str(limit), "hd_strategy": hd_strategy}),
        ("multipart", {"image_file": ("image.png", img_bytes, "image/png"), "mask_file": ("mask.png", mask_bytes, "image/png")}, {"model": model, "size_limit": str(limit), "hd_strategy": hd_strategy}),
    ]

    for url in candidates:
        for mode, files, data in payloads:
            try:
                if mode == "json":
                    r = requests.post(url, json=data, timeout=180)
                elif mode == "form":
                    r = requests.post(url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=180)
                else:
                    r = requests.post(url, files=files, data=data, timeout=180)

                if r.status_code == 200:
                    _LAMA_FAILS = 0
                    ct = (r.headers.get("content-type") or "").lower()
                    if "application/json" in ct:
                        j = r.json()
                        b64 = j.get("image") or j.get("data") or j.get("result") or ""
                        out_crop = _decode_json_image(b64)
                    else:
                        out_crop = Image.open(_io.BytesIO(r.content)).convert("RGBA")

                    base_img = img.convert("RGBA").copy()
                    base_img.paste(out_crop, (x1, y1), out_crop)
                    return base_img
                else:
                    _LAMA_FAILS += 1
            except Exception:
                _LAMA_FAILS += 1
                continue
    return None

# ================= OpenCV inpaint (fallback) =================
def _opencv_inpaint(img: Image.Image, mask_u8: np.ndarray, box: Tuple[int, int, int, int]):
    if not _HAVE_CV or mask_u8 is None:
        return None
    arr = np.array(img.convert("RGBA"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    _, _, w, h = box
    r1 = int(max(3, min(20, 0.06 * min(w, h))))
    r2 = int(max(r1 + 2, 0.08 * min(w, h)))
    out1 = cv2.inpaint(bgr, mask_u8, r1, cv2.INPAINT_TELEA)
    mask2 = cv2.dilate(mask_u8, np.ones((3, 3), np.uint8), iterations=1)
    out2 = cv2.inpaint(out1, mask2, r2, cv2.INPAINT_NS)
    return Image.fromarray(cv2.cvtColor(out2, cv2.COLOR_BGR2RGBA))

def _add_soft_grain(target: Image.Image, mask_l: Image.Image, strength: int = 3):
    if strength <= 0: return
    W, H = target.size
    base = np.array(target, dtype=np.int16)
    alpha = np.array(mask_l, dtype=np.uint8)[..., None]
    noise = np.random.normal(0, strength, (H, W, 1)).astype(np.int16)
    rnd = np.clip(base[..., :3] + noise, 0, 255).astype(np.uint8)
    out = base.copy()
    out[..., :3] = np.where(alpha > 0, ((0.8 * base[..., :3] + 0.2 * rnd)).astype(np.uint8), base[..., :3])
    target.paste(Image.fromarray(out.astype(np.uint8)), (0, 0))

# ================= Public eraser =================
def _apply_directional_blur_region(base: Image.Image, mask_l: Image.Image,
                                   k: int = 15, vertical: bool = True, alpha: float = 0.4):
    if not _HAVE_CV or k <= 1: return
    arr = np.array(base.convert("RGBA"))
    rgb = arr[..., :3]
    import cv2 as _cv2
    blurred = _cv2.blur(rgb, (1, int(k)) if vertical else (int(k), 1))
    m = np.array(mask_l, dtype=np.float32) / 255.0
    m = _cv2.GaussianBlur(m, (0, 0), 1.0)[..., None] * alpha
    out = (rgb * (1 - m) + blurred * m).astype(np.uint8)
    arr[..., :3] = out
    base.paste(Image.fromarray(arr), (0, 0))

# ----- Brush-mask based inpaint -----
def inpaint_with_mask(img: Image.Image, mask_u8: np.ndarray, box_hint: Optional[Tuple[int,int,int,int]]=None) -> Optional[Image.Image]:
    if box_hint is None:
        ys, xs = np.where(mask_u8 > 0)
        if len(xs)==0 or len(ys)==0:
            return None
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
        box_hint = (int(x1), int(y1), int(x2-x1), int(y2-y1))

    before = img.copy()
    out = _inpaint_http(img, mask_u8)
    if out is not None:
        if _diff_score(before, out, box_hint) < 2.0:
            ocv = _opencv_inpaint(img, mask_u8, box_hint)
            return ocv if ocv is not None else out
        return out

    ocv = _opencv_inpaint(img, mask_u8, box_hint)
    return ocv

def erase_text_area(img: Image.Image,
                    box: Tuple[int, int, int, int],
                    expand: int = 14,
                    feather: int = 3,
                    corner: int = 6,
                    aggressive: int = 2,
                    mask_mode: str = "rect",
                    pattern: str | None = None) -> None:
    if mask_mode == "rect":
        mask_full = np.array(_rect_mask(img.size, box, feather=0, corner=corner), dtype=np.uint8)
    elif mask_mode == "text":
        mask_full = _build_text_mask(img, box, expand_px=expand, grow_px=4 if aggressive else 0)
    else:
        mask_full = _build_text_mask(img, box, expand_px=expand, grow_px=4 if aggressive else 0)
        if mask_full is not None and _mask_coverage(mask_full, box) < 0.35:
            rect_retry = np.array(_rect_mask(img.size, box, feather=0, corner=corner), dtype=np.uint8)
        else:
            rect_retry = None

    before = img.copy()
    out = _inpaint_http(img, mask_full) if mask_full is not None else None
    if out is not None:
        changed = _diff_score(before, out, box)
        if aggressive >= 2 and changed < 4.0:
            if mask_mode != "rect":
                rect_m = rect_retry if 'rect_retry' in locals() and rect_retry is not None \
                         else np.array(_rect_mask(img.size, box, feather=0, corner=corner), dtype=np.uint8)
                out2 = _inpaint_http(img, rect_m)
                if out2 is not None:
                    img.paste(out2); return
        img.paste(out); return

    out2 = _opencv_inpaint(img, mask_full, box) if mask_full is not None else None
    if out2 is not None:
        img.paste(out2); return

    x, y, w, h = box
    bg = estimate_bg_color(img, (x, y, w, h), expand=expand)
    fill_layer = Image.new("RGBA", img.size, bg + (255,))
    mask_l = _rect_mask(img.size, box, feather=feather, corner=corner)
    img.alpha_composite(fill_layer, (0, 0), mask_l)
    if pattern in ("vertical", "horizontal"):
        k = max(9, (h if pattern == "vertical" else w) // 4)
        _apply_directional_blur_region(img, mask_l, k=k, vertical=(pattern == "vertical"))
    try:
        _add_soft_grain(img, mask_l, strength=4)
    except Exception:
        pass

# ================= Font fit & soft rect =================
def draw_soft_rect(draw: ImageDraw.ImageDraw,
                   img: Image.Image,
                   box: Tuple[int, int, int, int],
                   color: Tuple[int, int, int] | Tuple[int, int, int, int],
                   blur_radius: int = 2,
                   corner: int = 6) -> None:
    x, y, w, h = box
    rect = Image.new("RGBA", img.size, (0, 0, 0, 0))
    rdraw = ImageDraw.Draw(rect)
    fill_color = color + (255,) if isinstance(color, tuple) and len(color) == 3 else color
    rdraw.rounded_rectangle([x, y, x + w, y + h], radius=corner, fill=fill_color)
    blurred = rect.filter(ImageFilter.GaussianBlur(blur_radius))
    img.alpha_composite(blurred)

def fit_font(font_path: str,
             text: str,
             box_w: int,
             box_h: int,
             max_size: int = 44,
             min_size: int = 14,
             line_width: int = 22) -> Tuple[ImageFont.FreeTypeFont, str]:
    import textwrap
    tmp = Image.new("RGBA", (max(1, box_w), max(1, box_h)))
    draw = ImageDraw.Draw(tmp)

    def wrap_for_font(fnt: ImageFont.FreeTypeFont):
        best = None
        for ch in range(max(8, int(line_width*0.6)), max(8, int(line_width*2))+1, 2):
            cand = textwrap.fill(text, width=ch)
            bbox = draw.multiline_textbbox((0, 0), cand, font=fnt, spacing=2, align="center")
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if tw <= box_w and th <= box_h:
                best = (cand, tw, th)
        if best: return best[0], best[1], best[2]

        tight = None
        for ch in range(max(8, int(line_width*0.6)), max(8, int(line_width*2))+1, 2):
            cand = textwrap.fill(text, width=ch)
            bbox = draw.multiline_textbbox((0, 0), cand, font=fnt, spacing=2, align="center")
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            score = max(0, tw - box_w) + max(0, th - box_h)
            if tight is None or score < tight[0]:
                tight = (score, cand, tw, th)
        return tight[1], tight[2], tight[3]

    for size in range(max_size, min_size - 1, -1):
        try:
            f = ImageFont.truetype(font_path, size)
        except Exception:
            continue
        wrapped, tw, th = wrap_for_font(f)
        if tw <= box_w - 4 and th <= box_h - 4:
            return f, wrapped

    f = ImageFont.truetype(font_path, min_size)
    wrapped, _, _ = wrap_for_font(f)
    return f, wrapped
