# translate_lingva.py
from __future__ import annotations
from typing import Optional
import os, time, random, sqlite3, requests, string
from dotenv import load_dotenv
from glossary import GLOSSARY_MAP, KEEP_AS_IS  # KEEP_AS_IS-ийг шууд ашиглая

load_dotenv()  # .env-г уншина (ажлын директорт байх ёстой)

from style import (
    apply_glossary_tokenwise,
    is_sfx, translate_sfx,
    reflow_fragments, post_polish_mn
)
from config import SOURCE_LANG, TARGET_LANG

# -------- ENV --------
SRC_DEFAULT = (os.getenv("SOURCE_LANG") or SOURCE_LANG or "auto").strip()
TGT_DEFAULT = (os.getenv("TARGET_LANG") or TARGET_LANG or "mn").strip()
TM_DB = os.getenv("TM_DB", "tm.db")

# API KEY – хэд хэдэн нэршлээс хайна
API_KEY = (
    os.getenv("LINGVA_API_KEY")
    or os.getenv("LINGVANEX_API_KEY")
    or os.getenv("LINGVA")  # fallback
)

BASE_HOST = (os.getenv("LINGVANEX_API_HOST") or "https://api-b2b.backenster.com").rstrip("/")
TRANSLATE_URL = f"{BASE_HOST}/b1/api/v3/translate"

TIMEOUT_S = float(os.getenv("LINGVA_TIMEOUT", "10"))
MAX_TRIES = int(os.getenv("LINGVA_MAX_ATTEMPTS", "3"))
BACKOFF   = float(os.getenv("LINGVA_BACKOFF_BASE", "0.5"))

# Санity log: эхлэхдээ нэг удаа хэвлэнэ
print(f"[lingva:init] host={BASE_HOST}  key_loaded={bool(API_KEY)}")

# Хэрвээ KEY байхгүй бол шууд ойлгомжтой алдаа шиднэ
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
        # source == target бол TM бохирдуулж, дараа нь кэш “англиар”-аа эргэн ирэх эрсдэлтэй — хадгалахгүй
        if target == source:
            return
        c = _tm_connect()
        c.execute("INSERT OR REPLACE INTO translations VALUES (?,?,?,?)",
                  (source, target, model, int(time.time())))
        c.commit(); c.close()
    except Exception:
        pass


# -------- Glossary utils --------
# Unicode punctuation нэмэлт
PUNCT_EXTRA = "…“”‘’—–-"
PUNCT = set(string.punctuation + PUNCT_EXTRA)

# Punctuation-гүй lowercase түлхүүр бүхий glossary
_NORMALIZED_GLOSSARY = {}
for _k, _v in GLOSSARY_MAP.items():
    if not _k:
        continue
    nk = "".join(ch for ch in _k.lower() if ch not in PUNCT).strip()
    if nk:
        _NORMALIZED_GLOSSARY[nk] = _v

# --- normalization helpers for KEEP/Glossary ---
def _normalize_token(s: str) -> str:
    return "".join(ch for ch in s.strip().lower()
                   if ch not in PUNCT and not ch.isspace())

# All normalized glossary keys (to avoid KEEP clashes)
_GLOSS_NORM_KEYS = {
    _normalize_token(k) for k in (GLOSSARY_MAP or {}).keys() if k
}

# Normalized KEEP_AS_IS, but exclude anything that overlaps with the glossary
_KEEP_NORM = {
    _normalize_token(t)
    for t in (KEEP_AS_IS or set())
    if _normalize_token(t) and _normalize_token(t) not in _GLOSS_NORM_KEYS
}

def apply_glossary(text: str) -> Optional[str]:
    """
    Текстийн захын punctuation-ыг салгаж, цөм үгийг glossary-д тааруулна.
    Таарвал анхны punctuation-аа хадгалж буцаана. (давхар punctuation-ыг бас шүүх)
    """
    if not text or not text.strip():
        return None

    s = text
    # leading/trailing punctuation ялгах
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

    # 1) Шууд тааруулалт
    mapped = GLOSSARY_MAP.get(key)
    if mapped is None:
        # 2) Punctuation-гүй тааруулалт
        norm_key = "".join(ch for ch in key if ch not in PUNCT).strip()
        if norm_key:
            mapped = _NORMALIZED_GLOSSARY.get(norm_key)

    if mapped is None:
        return None

    # Давхардсан төгсгөлийн punctuation-ыг шахах (ж: "аа?" + "?" -> "аа?")
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
    results = [None] * len(raw)

    for i, s in enumerate(raw):
        if not s:
            results[i] = ""
            continue

        # 0) SFX
        if is_sfx(s):
            val = translate_sfx(s)
            results[i] = post_polish_mn(val) if do_polish else val
            tm_put(s, results[i], "sfx")
            continue

        # 1) SFX handled above

        # 2) Glossary (punct-aware) FIRST
        g = apply_glossary(s)
        if g is not None:
            out = post_polish_mn(g) if do_polish else g
            results[i] = out
            tm_put(s, out, "glossary")
            continue

        # 3) KEEP_AS_IS — only keep those that don't collide with glossary
        if _normalize_token(s) in _KEEP_NORM:
            results[i] = s
            tm_put(s, s, "keep")
            continue

        # 3) TM cache (source-той адил биш үед л ашиглана)
        cached = None if force_online else tm_get(s)
        if cached is not None and cached != s:
            results[i] = cached
            continue

        # 4) Онлайн / локал fallback
        if local_only:
            val = apply_glossary_tokenwise(s)
            results[i] = post_polish_mn(val) if do_polish else val
            continue

        tr = lingva_translate_with_retries(s, source_lang, target_lang)
        if tr:
            final = apply_glossary_tokenwise(tr)
            final = post_polish_mn(final) if do_polish else final
            results[i] = final
            tm_put(s, final, "lingva")
        else:
            val = apply_glossary_tokenwise(s)
            final = post_polish_mn(val) if do_polish else val
            results[i] = final
            if final != s:
                tm_put(s, final, "local-fallback")

    # >>> ХАМГИЙН ЧУХАЛ: үргэлж жагсаалт буцаа <<<
    return results

def translate_lines(lines, source_lang: str = SRC_DEFAULT, target_lang: str = TGT_DEFAULT,
                    reflow=False, do_polish=True, system_override: Optional[str]=None):
    out = translate_lines_impl(lines, source_lang, target_lang, reflow, do_polish,
                               system_override, local_only=False, force_online=False)
    return out if isinstance(out, list) else []
