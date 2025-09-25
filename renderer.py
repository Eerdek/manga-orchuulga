# renderer.py
from __future__ import annotations
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
from utils import OCRBox, fit_font
import os, pathlib

def _load_image(src: str | Image.Image) -> Image.Image:
    if isinstance(src, Image.Image):
        return src.convert("RGBA")
    from PIL import Image as _Img
    return _Img.open(src).convert("RGBA")

# ---- Font resolver: DEFAULT_FONT_PATH -> system fallbacks ----
_CANDIDATES = [
    # env/config өгөгдвөл тэрийгээ эхэлж хэрэглэнэ
    os.getenv("DEFAULT_FONT_PATH") or "",
    # Windows
    r"C:\Windows\Fonts\NotoSans-Medium.ttf",
    r"C:\Windows\Fonts\NotoSans-Regular.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/NotoSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    # Linux
    "/usr/share/fonts/truetype/noto/NotoSans-Medium.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

def _pick_font(path_hint: str | None, size: int) -> ImageFont.FreeTypeFont:
    tried = []
    if path_hint:
        try:
            return ImageFont.truetype(path_hint, size)
        except Exception:
            tried.append(path_hint)
    for p in _CANDIDATES:
        if not p: continue
        if not pathlib.Path(p).exists(): continue
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            tried.append(p)
            continue
    # эцсийн fallback
    return ImageFont.load_default()

def render_translations(
    src_img: str | Image.Image,
    boxes: List[OCRBox],
    translations: List[str],
    default_font_path: str,
    **kwargs,  # үл мэдэгдэх параметрүүдийг зүгээр л үл тооно
) -> Image.Image:
    styles = [{"fontSize": None, "color": None} for _ in boxes]
    return render_translations_styled(src_img, boxes, translations, styles, default_font_path)

def render_translations_styled(
    src_img: str | Image.Image,
    boxes: List[OCRBox],
    translations: List[str],
    styles: List[Dict[str, Any]],
    default_font_path: str,
) -> Image.Image:
    """
    Цагаан background ЗАВСРАЛГҮЙ.
    Текстийг box дотор яг төвд нь байрлуулна.
    Фонт: styles[i].fontSize байвал тэр, үгүй бол автоматаар fit.
    """
    img = _load_image(src_img)
    draw = ImageDraw.Draw(img)

    for i, b in enumerate(boxes):
        text = (translations[i] if i < len(translations) else "") or ""
        if not text.strip():
            continue

        x, y, w, h = map(int, b.bbox)
        st = styles[i] if i < len(styles) and isinstance(styles[i], dict) else {}
        fs = int(st.get("fontSize") or 0)
        color = st.get("color") or "#111111"

        if fs > 0:
            fnt = _pick_font(default_font_path, fs)
            wrapped = text
            # хэрвээ багтахгүй бол багтаах хүртэл автоматаар багасгана
            bbox = draw.multiline_textbbox((0, 0), wrapped, font=fnt, spacing=2, align="center")
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            if tw > w or th > h:
                fnt, wrapped = fit_font(
                    default_font_path or "",
                    text, w, h,
                    max_size=fs, min_size=12,
                )
        else:
            fnt, wrapped = fit_font(
                default_font_path or "",
                text, w, h,
                max_size=44, min_size=12,
            )
            # fit_font дотор truetype нээгдэхгүй бол өөрөө сонгоно
            if isinstance(fnt, ImageFont.FreeTypeFont) is False:
                fnt = _pick_font(default_font_path, 24)

        # яг төв байрлал
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=fnt, spacing=2, align="center")
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        tx = x + max(0, (w - tw) // 2)
        ty = y + max(0, (h - th) // 2)

        draw.multiline_text((tx, ty), wrapped, font=fnt, fill=color, align="center", spacing=2)

    return img
