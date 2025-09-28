from __future__ import annotations
from typing import Optional, List, Tuple
import os, time, random, sqlite3, requests, string, re
from dotenv import load_dotenv

# --- glossary импорт (KEEP_PATTERNS байхгүй бол аюулгүй fallback) ---
try:
    from glossary import GLOSSARY_MAP, KEEP_AS_IS, KEEP_PATTERNS  # noqa
except Exception:
    from glossary import GLOSSARY_MAP, KEEP_AS_IS  # type: ignore
    KEEP_PATTERNS: List[str] = []

load_dotenv()

# --- style импортыг уян хатан ---
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
    # эх хэлний боломжит хувилбарууд (давхардалгүй)
    cand = []
    for s in ["auto", (src or "").lower(), "ja", "en", "zh", "ko"]:
        if s and s not in cand:
            cand.append(s)
    for s in cand:
        last = None
        for attempt in range(1, MAX_TRIES + 1):
            out = _lingva_call(text, s, tgt)
            if out:
                return out
            if attempt < MAX_TRIES:
                wait = BACKOFF * (2 ** (attempt - 1))
                time.sleep(wait * (0.9 + 0.2 * random.random()))
    return None

# -------- Fallbacks for style.* --------
def _fallback_apply_glossary_tokenwise(text: str) -> str:
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
    text = " ".join([l.strip() for l in lines if l.strip()])
    if not text:
        return lines
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
_TERMINAL_RE = re.compile(r'[.!?。！？…]$')

def _has_latin(s: str) -> bool:
    return bool(_LATIN_RE.search(s or ""))

def _same_ignorecase(a: str, b: str) -> bool:
    return (a or "").strip().lower() == (b or "").strip().lower()

# -------- KEEP masking (орчуулагдахгүй хэсгийг хамгаална) --------
_PLACEHOLDER = "⟦K{}⟧"

def _build_keep_regex():
    toks = [re.escape(t) for t in (KEEP_AS_IS or set()) if t]
    pats = list(KEEP_PATTERNS or [])
    if not toks and not pats:
        return None
    alt = []
    if toks:
        alt.append(r"(?:\b(?:%s)\b)" % "|".join(toks))
    if pats:
        alt.extend(pats)
    rx = "|".join(alt)
    try:
        return re.compile(rx, flags=re.IGNORECASE)
    except re.error:
        return None

_KEEP_RX = _build_keep_regex()

def _mask_keep_tokens(text: str):
    if not text or not _KEEP_RX:
        return text, []
    vals: List[str] = []
    def repl(m):
        i = len(vals)
        vals.append(m.group(0))
        return _PLACEHOLDER.format(i)
    masked = _KEEP_RX.sub(repl, text)
    return masked, vals

def _unmask_keep_tokens(text: str, vals: List[str]):
    if not text or not vals:
        return text
    for i, v in enumerate(vals):
        text = text.replace(_PLACEHOLDER.format(i), v)
    return text

# -------- Geometry-based grouping --------
def _coalesce_by_geometry(lines: List[str],
                          boxes: List[Tuple[int,int,int,int]]) -> Tuple[List[str], List[List[int]]]:
    """
    Нэг бөмбөлөгт хамт байрласан OCR мөрүүдийг bbox-аар бүлэглэнэ.
    Буцах:
      merged_texts, groups (эх индексүүдийн жагсаалт)
    """
    n = len(lines)
    idxs = list(range(n))

    # уншлагын дараалал: эхлээд y, тэгээд x
    idxs.sort(key=lambda i: (boxes[i][1], boxes[i][0]))

    merged: List[str] = []
    groups: List[List[int]] = []

    def same_bubble(i: int, j: int) -> bool:
        x1,y1,w1,h1 = boxes[i]; x2,y2,w2,h2 = boxes[j]
        r1 = (x1, y1, x1+w1, y1+h1)
        r2 = (x2, y2, x2+w2, y2+h2)

        # хэвтээ давхцлын хувь (intersection / min width)
        ix = min(r1[2], r2[2]) - max(r1[0], r2[0])
        horiz_overlap = max(0, ix) / max(1, min(w1, w2))

        # босоо зай
        vgap = max(0, y2 - r1[3]) if y2 >= y1 else max(0, y1 - r2[3])

        # төвүүдийн x зөрүү
        cx1, cx2 = x1 + w1/2.0, x2 + w2/2.0
        dx = abs(cx1 - cx2)

        h_avg = (h1 + h2) / 2.0
        w_avg = (w1 + w2) / 2.0

        return (horiz_overlap >= 0.35) and (vgap <= max(10, 0.7*h_avg)) and (dx <= 0.6*w_avg)

    used = set()
    for i in idxs:
        if i in used:
            continue
        grp = [i]
        text = (lines[i] or "").strip()
        used.add(i)

        # ихдээ дараагийн 2-3 мөрийг л наана
        added = 0
        for j in idxs:
            if j in used:
                continue
            if added >= 3:
                break
            if same_bubble(grp[-1], j):
                t = (lines[j] or "").strip()
                if t:
                    text = f"{text} {t}"
                grp.append(j)
                used.add(j)
                added += 1

        merged.append(text)
        groups.append(sorted(grp))

    zipped = sorted(zip(groups, merged), key=lambda gm: min(gm[0]) if gm[0] else 1e9)
    groups = [g for g,_ in zipped]
    merged = [m for _,m in zipped]
    return merged, groups

# --- Text-only fallback grouping ---
def _is_short_shout(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return False
    core = re.sub(r"[^\w]+", "", t)
    if len(core) <= 8 and (core.isupper() or core.lower() in {
        "hey","honey","this","clearly","wait","look","oh","ah","um","hmm","yo"
    }):
        return True
    if re.search(r'[,:;–—-]$', t):
        return True
    return False

def _should_merge_text(cur: str, nxt: str) -> bool:
    if not cur or not nxt:
        return False
    a = cur.strip(); b = nxt.strip()
    if not a or not b:
        return False
    if not _TERMINAL_RE.search(a):
        if _is_short_shout(a) or len(a) <= 18:
            return True
    if _is_short_shout(a) and len(b) >= 2:
        return True
    return False

def _coalesce_by_text(lines: List[str]) -> Tuple[List[str], List[List[int]]]:
    merged: List[str] = []
    groups: List[List[int]] = []
    i, n = 0, len(lines)
    while i < n:
        cur = (lines[i] or "").strip()
        if not cur:
            merged.append("")
            groups.append([i])
            i += 1
            continue
        grp = [i]; text = cur; extra = 0; j = i+1
        while j < n and extra < 2:
            nxt = (lines[j] or "").strip()
            if not nxt:
                break
            if _should_merge_text(text, nxt):
                text = f"{text} {nxt}"; grp.append(j); extra += 1; j += 1
                if _TERMINAL_RE.search(text.strip()):
                    break
            else:
                break
        merged.append(text); groups.append(grp); i = grp[-1] + 1
    return merged, groups

# -------- Custom semantic rules (optional) --------
def apply_semantic_rules(text: str) -> str:
    if not text:
        return text
    # Жишээ дүрэм:
    text = re.sub(r"\binput text\b", "энэ юу юм бэ", text, flags=re.IGNORECASE)
    return text

# -------- Public API --------
def translate_lines_impl(lines,
                         source_lang: str = SRC_DEFAULT,
                         target_lang: str = TGT_DEFAULT,
                         reflow=False, do_polish=True,
                         system_override: Optional[str]=None,
                         local_only=False, force_online=False,
                         boxes: Optional[List[Tuple[int,int,int,int]]] = None):
    if not lines:
        return []

    out = []
    for src_text in lines:
        src_text = (src_text or "").strip()
        if not src_text:
            out.append("")
            continue

        # 1. Glossary
        g = apply_glossary(src_text)
        if g is not None:
            result = post_polish_mn(g) if do_polish else g
            out.append(result)
            tm_put(src_text, result, "glossary")
            continue

        # 2. KEEP (хэрвээ яг хэвээр үлдээх ёстой бол)
        norm = _normalize_token(src_text)
        if norm in _KEEP_NORM:
            out.append(src_text)
            tm_put(src_text, src_text, "keep")
            continue

        # 3. SFX
        if is_sfx(src_text):
            val = translate_sfx(src_text)
            result = post_polish_mn(val) if do_polish else val
            out.append(result)
            tm_put(src_text, result, "sfx")
            continue

        # 4. TM cache (хуучин орчуулга байгаа бол)
        if not force_online:
            cached = tm_get(src_text)
            if cached is not None and cached != src_text:
                out.append(cached)
                continue

        # 5. Local-only (offline glossary-based fallback)
        if local_only:
            val = apply_glossary_tokenwise(src_text)
            result = post_polish_mn(val) if do_polish else val
            out.append(result)
            continue

        # 6. KEEP токенуудыг масклах
        masked_text, keep_vals = _mask_keep_tokens(src_text)

        # 7. API-р орчуулах
        tr = lingva_translate_with_retries(masked_text, source_lang, target_lang)
        if not tr or _same_ignorecase(tr, masked_text):
            for s in ["en", "auto", "ja"]:
                tr2 = lingva_translate_with_retries(masked_text, s, target_lang)
                if tr2 and not _same_ignorecase(tr2, masked_text):
                    tr = tr2
                    break

        # Хэрвээ API-гийн орчуулга амжилттай бол
        if tr:
            tr = _unmask_keep_tokens(tr, keep_vals)          # KEEP-г яг хэлбэрээр нь сэргээх
            final = apply_glossary_tokenwise(tr)
            final = apply_semantic_rules(final)
            final = post_polish_mn(final) if do_polish else final
            out.append(final)
            tm_put(src_text, final, "lingva")
        else:
            # Орчуулга амжилтгүй бол (API-гаас буцаасан текст англи хэвээр бол)
            val = apply_glossary_tokenwise(src_text)
            final = post_polish_mn(val) if do_polish else val
            out.append(final)
            if final != src_text:
                tm_put(src_text, final, "local-fallback")

    return out

def translate_lines(lines, source_lang: str = SRC_DEFAULT, target_lang: str = TGT_DEFAULT,
                    reflow=False, do_polish=True, system_override: Optional[str]=None):
    out = translate_lines_impl(lines, source_lang, target_lang, reflow, do_polish,
                               system_override, local_only=False, force_online=False)
    return out if isinstance(out, list) else []