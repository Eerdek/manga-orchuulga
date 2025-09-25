# translate_lingva.py
from __future__ import annotations
from typing import Optional
import os, time, random, sqlite3, requests, string, re
from dotenv import load_dotenv
from glossary import GLOSSARY_MAP, KEEP_AS_IS  # SFX_MAP хэрэггүй

load_dotenv()

# --- style импортыг уян хатан болгоё (байхгүй бол fallback-ууд ажиллана)
try:
    from style import (
        apply_glossary_tokenwise as _apply_glossary_tokenwise_ext,
        is_sfx as _is_sfx_ext,
        translate_sfx as _translate_sfx_ext,
        reflow_fragments as _reflow_fragments_ext,
        post_polish_mn as _post_polish_mn_ext,
    )
except Exception:
    _apply_glossary_tokenwise_ext = None
    _is_sfx_ext = None
    _translate_sfx_ext = None
    _reflow_fragments_ext = None
    _post_polish_mn_ext = None

from config import SOURCE_LANG, TARGET_LANG

# -------- ENV --------
SRC_DEFAULT = (os.getenv("SOURCE_LANG") or SOURCE_LANG or "auto").strip()
TGT_DEFAULT = (os.getenv("TARGET_LANG") or TARGET_LANG or "mn").strip()
TM_DB = os.getenv("TM_DB", "tm.db")

API_KEY = (
    os.getenv("LINGVA_API_KEY")
    or os.getenv("LINGVANEX_API_KEY")
    or os.getenv("LINGVA")
)

BASE_HOST = (os.getenv("LINGVANEX_API_HOST") or "https://api-b2b.backenster.com").rstrip("/")
TRANSLATE_URL = f"{BASE_HOST}/b1/api/v3/translate"

TIMEOUT_S = float(os.getenv("LINGVA_TIMEOUT", "10"))
MAX_TRIES = int(os.getenv("LINGVA_MAX_ATTEMPTS", "3"))
BACKOFF   = float(os.getenv("LINGVA_BACKOFF_BASE", "0.5"))

print(f"[lingva:init] host={BASE_HOST}  key_loaded={bool(API_KEY)}")

if not API_KEY:
    raise RuntimeError(
        "LINGVA_API_KEY (эсвэл LINGVANEX_API_KEY) олдсонгүй. "
        "Түлхүүрээ .env-д LINGVA_API_KEY=... гэж тавиад, main.py-г тэр хавтаснаас нь ажиллуул."
    )

# -------- Tiny TM --------
def _tm_connect():
    conn = sqlite3.connect(TM_DB, timeout=5)
    conn.execute("""CREATE TABLE IF NOT EXISTS translations(
        source TEXT PRIMARY KEY,
        target TEXT,
        model  TEXT,
        ts     INTEGER
    )""")
    conn.commit()
    return conn

def tm_get(source: str) -> Optional[str]:
    try:
        c = _tm_connect()
        row = c.execute("SELECT target FROM translations WHERE source=?", (source,)).fetchone()
        c.close()
        return row[0] if row else None
    except Exception:
        return None

def tm_put(source: str, target: str, model: str = "lingva"):
    try:
        if target == source:
            return
        c = _tm_connect()
        c.execute("INSERT OR REPLACE INTO translations VALUES (?,?,?,?)",
                  (source, target, model, int(time.time())))
        c.commit(); c.close()
    except Exception:
        pass

# -------- Glossary utils --------
PUNCT_EXTRA = "…“”‘’—–-"
PUNCT = set(string.punctuation + PUNCT_EXTRA)

_NORMALIZED_GLOSSARY = {}
for _k, _v in GLOSSARY_MAP.items():
    if not _k:
        continue
    nk = "".join(ch for ch in _k.lower() if ch not in PUNCT).strip()
    if nk:
        _NORMALIZED_GLOSSARY[nk] = _v

def _normalize_token(s: str) -> str:
    return "".join(ch for ch in (s or "").strip().lower()
                   if ch not in PUNCT and not ch.isspace())

_GLOSS_NORM_KEYS = {_normalize_token(k) for k in (GLOSSARY_MAP or {}).keys() if k}

_KEEP_NORM = {
    _normalize_token(t)
    for t in (KEEP_AS_IS or set())
    if _normalize_token(t) and _normalize_token(t) not in _GLOSS_NORM_KEYS
}

def apply_glossary(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None

    s = text
    i, j = 0, len(s) - 1
    while i <= j and s[i] in PUNCT:
        i += 1
    while j >= i and s[j] in PUNCT:
        j -= 1

    core = s[i:j+1]
    lead = s[:i]
    tail = s[j+1:]

    if not core or not core.strip():
        return None

    key = core.lower().strip()

    mapped = GLOSSARY_MAP.get(key)
    if mapped is None:
        norm_key = "".join(ch for ch in key if ch not in PUNCT).strip()
        if norm_key:
            mapped = _NORMALIZED_GLOSSARY.get(norm_key)

    if mapped is None:
        return None

    if tail and mapped and mapped.endswith(tail[:1]) and mapped[-1] in PUNCT:
        tail = tail[1:]

    return f"{lead}{mapped}{tail}"

# -------- Low-level call --------
def _lingva_call(text: str, src: str, tgt: str) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {"platform": "api", "to": tgt, "data": text}
    if src and src.lower() != "auto":
        payload["from"] = src

    try:
        r = requests.post(TRANSLATE_URL, json=payload, headers=headers, timeout=TIMEOUT_S)
        print("[lingva] status:", r.status_code, "preview:", (r.text or "")[:120])
    except requests.RequestException as e:
        print("[lingva] request error:", e)
        return None

    try:
        j = r.json()
        if isinstance(j, dict):
            res = j.get("result") or j.get("translation") or j.get("text")
            if isinstance(res, str) and res.strip():
                return res.strip()
    except Exception:
        pass

    if 200 <= r.status_code < 300 and isinstance(r.text, str) and r.text.strip():
        return r.text.strip()

    return None

def lingva_translate_with_retries(text: str, src: str, tgt: str) -> Optional[str]:
    last = None
    for attempt in range(1, MAX_TRIES+1):
        out = _lingva_call(text, src, tgt)
        if out:
            return out
        if attempt < MAX_TRIES:
            wait = BACKOFF * (2 ** (attempt-1))
            time.sleep(wait * (0.9 + 0.2*random.random()))
    return last

# -------- Fallbacks for style.* if missing --------
def _fallback_apply_glossary_tokenwise(text: str) -> str:
    # энгийн, кейсийн эхний үсгийг хадгална
    def repl(m):
        src = m.group(0)
        key = src.lower()
        tgt = GLOSSARY_MAP.get(key)
        if not tgt:
            return src
        return (tgt[:1].upper() + tgt[1:]) if src[:1].isupper() else tgt
    vocab = sorted(GLOSSARY_MAP.keys(), key=len, reverse=True)
    if not vocab:
        return text
    pattern = r'\b(' + '|'.join(re.escape(v) for v in vocab) + r')\b'
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)

def _fallback_post_polish_mn(text: str) -> str:
    t = (text or "").replace("...", "…")
    t = re.sub(r"\s+([,!?…])", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    if t and not t[0].isupper():
        t = t[0].upper() + t[1:]
    return t

def _fallback_reflow_fragments(lines):
    # OCR мөрүүдийг нэгтгэж нэгэн текст болгоно
    text = " ".join([l.strip() for l in lines if l.strip()])
    if not text:
        return lines
    import re
    # Өгүүлбэрээр таслана
    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def apply_glossary_tokenwise(text: str) -> str:
    return _apply_glossary_tokenwise_ext(text) if _apply_glossary_tokenwise_ext else _fallback_apply_glossary_tokenwise(text)

def post_polish_mn(text: str) -> str:
    return _post_polish_mn_ext(text) if _post_polish_mn_ext else _fallback_post_polish_mn(text)

def reflow_fragments(lines):
    return _reflow_fragments_ext(lines) if _reflow_fragments_ext else _fallback_reflow_fragments(lines)

def is_sfx(line: str) -> bool:
    return _is_sfx_ext(line) if _is_sfx_ext else False

def translate_sfx(line: str) -> str:
    return _translate_sfx_ext(line) if _translate_sfx_ext else line

# -------- Helpers --------
_LATIN_RE = re.compile(r'[A-Za-z]')

def _has_latin(s: str) -> bool:
    return bool(_LATIN_RE.search(s or ""))

def _same_ignorecase(a: str, b: str) -> bool:
    return (a or "").strip().lower() == (b or "").strip().lower()

# -------- Public API --------
def translate_lines_impl(lines,
                         source_lang: str = SRC_DEFAULT,
                         target_lang: str = TGT_DEFAULT,
                         reflow=False, do_polish=True,
                         system_override: Optional[str]=None,
                         local_only=False, force_online=False):
    if not lines:
        return []

    raw = reflow_fragments(lines) if reflow else lines
    results = []

    for s in raw:
        if not s:
            results.append("")
            continue

        norm = _normalize_token(s)

        # 1) Glossary (punct-aware) – ЭХЭНД
        g = apply_glossary(s)
        if g is not None:
            out = post_polish_mn(g) if do_polish else g
            results.append(out)
            tm_put(s, out, "glossary")
            continue

        # 2) KEEP_AS_IS – глоссариас давуу биш (давхцал шүүгдсэн)
        if norm in _KEEP_NORM:
            results.append(s)
            tm_put(s, s, "keep")
            continue

        # 3) SFX – зөвхөн style.is_sfx/translate_sfx
        if is_sfx(s):
            val = translate_sfx(s)
            out = post_polish_mn(val) if do_polish else val
            results.append(out)
            tm_put(s, out, "sfx")
            continue

        # 4) TM cache
        if not force_online:
            cached = tm_get(s)
            if cached is not None and cached != s:
                results.append(cached)
                continue

        # 5) Local-only fallback
        if local_only:
            val = apply_glossary_tokenwise(s)
            out = post_polish_mn(val) if do_polish else val
            results.append(out)
            continue

        # 6) Онлайн орчуулга
        tr = lingva_translate_with_retries(s, source_lang, target_lang)

        if (not tr or _same_ignorecase(tr, s)) and _has_latin(s):
            tr2 = lingva_translate_with_retries(s, "en", target_lang)
            if tr2 and not _same_ignorecase(tr2, s):
                print(f"[lingva:fallback-en] {s!r} -> {tr2!r}")
                tr = tr2

        if tr and not _same_ignorecase(tr, s):
            final = apply_glossary_tokenwise(tr)
            final = post_polish_mn(final) if do_polish else final
            results.append(final)
            tm_put(s, final, "lingva")
        else:
            # эцсийн fallback: токеноор глоссари
            val = apply_glossary_tokenwise(s)
            final = post_polish_mn(val) if do_polish else val
            results.append(final)
            if final != s:
                tm_put(s, final, "local-fallback")

    return results

def translate_lines(lines, source_lang: str = SRC_DEFAULT, target_lang: str = TGT_DEFAULT,
                    reflow=False, do_polish=True, system_override: Optional[str]=None):
    out = translate_lines_impl(lines, source_lang, target_lang, reflow, do_polish,
                               system_override, local_only=False, force_online=False)
    return out if isinstance(out, list) else []
