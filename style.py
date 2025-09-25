# style.py
import re
from glossary import KEEP_AS_IS, SFX_MAP  # Top-level дээр зөвхөн эдгээрийг импортолно

# --- SFX танилт хатуу горим ---
# Пунктуацын дараалал эсвэл зөвхөн SFX_MAP/KEEP_AS_IS-т буй КАПС ономатопеяг л SFX гэж үзнэ
_SFX_STRICT = {k.upper() for k in (set(SFX_MAP.keys()) | set(KEEP_AS_IS))}

def _punct_run(t: str) -> bool:
    # --, …, --- зэрэг цэвэр тэмдэгтийн мөр
    return bool(re.fullmatch(r"[.\-–—~…]{2,}", t or ""))

def is_sfx(line: str) -> bool:
    """
    ЗӨВХӨН:
    - цэвэр пунктуацын мөр (--, …, --- гэх мэт), эсвэл
    - SFX_STRICT доторх КАПС ономатопея байвал SFX гэж үзнэ.
    HONEY!, THIS, TODAY? гэх мэт энгийн КАПС үгс SFX БИШ.
    """
    t = (line or "").strip()
    if not t:
        return False
    if _punct_run(t) or t in {"--", "—"}:
        return True

    # төгсгөлийн пунктуацыг авч цөм үгийг шалгана
    core = re.sub(r"[.!?,;:~…]+$", "", t).strip()
    if not core:
        return False

    return core.isalpha() and core.upper() == core and core.upper() in _SFX_STRICT

def translate_sfx(line: str) -> str:
    """
    SFX_MAP-д байвал хөрвүүлж, KEEP_AS_IS-д байвал хэвээр үлдээнэ.
    Пунктуацын сүүлийг хадгална (ж: WHISPER... -> шивн-шивн…).
    """
    t = (line or "").strip()
    if _punct_run(t) or t in {"--", "—"}:
        return t  # ийм мөрүүдийг шууд үлдээнэ

    m = re.match(r"^(.*?)([.!?,;:~…]*)$", t)
    core, tail = m.group(1), m.group(2)
    up = core.upper()

    if up in SFX_MAP:
        return SFX_MAP[up] + tail
    if core in KEEP_AS_IS or up in KEEP_AS_IS:
        return core + tail

    return line  # ердийнхийг орхино (is_sfx False тул орчуулгад явна)

# --- Глоссари токен-байдлаар (word boundary) ---
def apply_glossary_tokenwise(text: str) -> str:
    """
    Үгийг зөвхөн бүтэн үгээр (word boundary) тааруулж сольдог — ‘hey’ ≠ ‘they’.
    Кейсийн эхний үсгийг хадгална.
    """
    from glossary import GLOSSARY_MAP  # function-level импорт (circular-оос сэргийлнэ)

    def repl(m: re.Match) -> str:
        src = m.group(0)
        key = src.strip().lower()
        tgt = GLOSSARY_MAP.get(key)
        if not tgt:
            return src
        # эхний үсгийн том/жижигийг хадгална
        if src[:1].isupper():
            return tgt[:1].upper() + tgt[1:]
        return tgt

    vocab = sorted(GLOSSARY_MAP.keys(), key=len, reverse=True)
    if not vocab:
        return text
    pattern = r'\b(' + '|'.join(re.escape(v) for v in vocab) + r')\b'
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)

# --- OCR фрагментуудыг нэгтгэх reflow ---
def reflow_fragments(lines):
    """
    OCR-аас ирсэн богино capitalized фрагментуудыг нэг өгүүлбэрт оруулах optional reflow.
      - <=12 тэмдэгт эсвэл connector-like бол богинод тооцоод буферт цуглуулна.
      - Төгсгөлийн цэг/асуулт/гансрал дээр эсвэл 40+ тэмдэгт хүрэхэд буферийг flush.
    """
    out = []
    buf = []

    def flush():
        if buf:
            out.append(" ".join(buf).strip())
            buf.clear()

    SHORT_LIMIT = 12
    CHAR_FLUSH_LIMIT = 40

    for s in lines:
        t = (s or "").strip()
        if not t:
            continue
        is_connector_like = t.endswith(("THE", "HE", "DO", "YOU", "WHEN")) or t.isupper()
        short = len(t) <= SHORT_LIMIT or is_connector_like

        if short and not t.endswith((".", "!", "?", "…")):
            buf.append(t)
            if len(" ".join(buf)) >= CHAR_FLUSH_LIMIT:
                flush()
        else:
            if buf:
                buf.append(t)
                flush()
            else:
                out.append(t)
    flush()
    return out

# --- Монгол текстийг бага зэргийн "balloon" өнгөлгөө ---
def post_polish_mn(text: str) -> str:
    """
    - гурван цэгийг юникод ‘…’ болгоно
    - punctuation-ийн өмнөх илүү зайг цэгцэлнэ
    - давхар зайг шахна
    - эхний үсгийг том болгоно (balloon стиль)
    """
    t = (text or "")
    t = t.replace("...", "…")
    t = re.sub(r"\s+([,!?…])", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t)
    if t and not t[0].isupper():
        t = t[0].upper() + t[1:]
    return t.strip()
