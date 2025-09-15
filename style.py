# style.py
import re
from glossary import GLOSSARY_MAP, KEEP_AS_IS, SFX_MAP

def normalize_for_lookup(s: str) -> str:
    return s.strip().lower()

def apply_glossary_tokenwise(text: str) -> str:
    """
    Үгийг “хоорондоо ялгаж” солих (word boundary) — ‘hey’ ≠ ‘they’
    Кейс-ийг аль болох хадгална.
    """
    def repl(m):
        src = m.group(0)
        key = normalize_for_lookup(src)
        tgt = GLOSSARY_MAP.get(key)
        if not tgt:
            return src
        # Preserve capitalization of first letter
        if src[:1].isupper():
            return tgt[:1].upper() + tgt[1:]
        return tgt

    # Build regex like r'\b(hey|honey|rintaro|...)\b' (case-insensitive)
    vocab = sorted(GLOSSARY_MAP.keys(), key=len, reverse=True)
    if not vocab:
        return text
    pattern = r'\b(' + '|'.join(re.escape(v) for v in vocab) + r')\b'
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)

def is_sfx(line: str) -> bool:
    """
    SFX гэж үзэх энгийн дүрэм: бүх үсэг нь латин, ихэнх нь CAPS, богинохон.
    """
    t = line.strip()
    if len(t) <= 12 and re.fullmatch(r"[A-Za-z\.\!\?']+", t or ""):
        caps_ratio = sum(ch.isupper() for ch in t if ch.isalpha()) / max(1, sum(ch.isalpha() for ch in t))
        return caps_ratio > 0.7
    return False

def translate_sfx(line: str) -> str:
    word = line.strip().strip(".!?:;")
    if word in KEEP_AS_IS:
        return word  # keep as is (e.g., TV)
    # Try SFX map (exact upper key)
    if word.upper() in SFX_MAP:
        return SFX_MAP[word.upper()]
    return line  # fallback: leave as is

def reflow_fragments(lines):
    """
    OCR-аас ирсэн богино capitalized фрагментуудыг нэг өгүүлбэрт оруулах optional reflow.
        Rule of thumb: short fragments are merged to form full phrases before translation.
        Improvements:
        - Treat fragments <= 12 chars as "short" (was 8).
        - Buffer consecutive short fragments and flush when hitting punctuation, connector conditions,
            or a character limit (default ~40 chars) to avoid creating overly long merged strings.
    """
    out = []
    buf = []
    def flush():
        if buf:
            out.append(" ".join(buf).strip())
            buf.clear()

    # parameters
    SHORT_LIMIT = 12
    CHAR_FLUSH_LIMIT = 40

    for s in lines:
        t = s.strip()
        if not t:
            continue
        is_connector_like = t.endswith(("THE", "HE", "DO", "YOU", "WHEN")) or t.isupper()
        short = len(t) <= SHORT_LIMIT or is_connector_like

        if short and not t.endswith((".", "!", "?", "…")):
            buf.append(t)
            # if buffered text is long enough, flush to avoid huge merges
            if len(" ".join(buf)) >= CHAR_FLUSH_LIMIT:
                flush()
        else:
            if buf:
                # append current (which may be end of sentence) to buffer then flush
                buf.append(t)
                flush()
            else:
                out.append(t)
    flush()
    return out

def post_polish_mn(text: str) -> str:
    t = text.replace("...", "…")
    t = re.sub(r"\s+([,!?…])", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t)
    # balloon style: first letter uppercase, rest as is
    if t and not t[0].isupper():
        t = t[0].upper() + t[1:]
    return t.strip()

