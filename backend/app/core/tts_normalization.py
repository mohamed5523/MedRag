"""
ULTIMATE Egyptian Arabic TTS Normalization
==========================================
Pipeline:
  1. Apply Egyptian mappings (MSA → Ammya)   ← wires egyptian_mappings.py
  2. Apply Egyptian phonology fixes
  3. Strip random LLM tashkeel
  4. Fix Ta Marbuta (ه → ة where needed)
  5. Fix Alef Maqsura confusion
  6. Normalize dates (weekdays → Egyptian)
  7. Normalize time formats
  8. Fix punctuation characters
  9. Normalize currency
 10. Ensure natural-pause punctuation
 11. Convert digits to Egyptian Arabic words
 12. Apply tactical tashkeel (pronunciation guides)
 13. Clean markdown / whitespace
"""

import re
from typing import Dict, Set


# ==========================================
# PRONUNCIATION DICTIONARIES
# ==========================================

# Western names whose ج is pronounced as the English "J" (soft jeem)
# Disabled English phonetic mapping (causes OpenAI TTS to switch to a foreign accent)
JEEM_NAMES: Dict[str, str] = {}

JEEM_WORDS = JEEM_NAMES

# ─── Tactical tashkeel ───────────────────────────────────────────────────────
# Key words that need vowel marks for the TTS engine to pronounce correctly.
# Keep this list targeted — over-tashkeel can confuse some engines.

# Disabled tactical tashkeel for OpenAI TTS (triggers foreign/Urdu accent with slang)
PRONUNCIATION_FIXES: Dict[str, str] = {}

# ==========================================
# NUMBER DEFINITIONS — EGYPTIAN STYLE
# ==========================================

ONES: Dict[int, str] = {
    0: "صفر",      1: "واحِد",      2: "إتنِين",   3: "تَلاتَة",
    4: "أَربَعة",  5: "خَمسَة",    6: "سِتَّة",   7: "سَبعَة",
    8: "تَمانيَة", 9: "تِسعَة",    10: "عَشَرة",  11: "حِداشَر",
    12: "إتناشَر", 13: "تَلاتاشَر", 14: "أَربَعتاشَر", 15: "خَمَستاشَر",
    16: "سِتَّاشَر", 17: "سَبَعتاشَر", 18: "تَمَنتاشَر", 19: "تِسَعتاشَر",
}

TENS: Dict[int, str] = {
    20: "عِشرين",  30: "تَلاتين",  40: "أَربَعين", 50: "خَمسين",
    60: "سِتِّين", 70: "سَبعين",   80: "تَمانين",  90: "تِسعين",
}

HUNDREDS: Dict[int, str] = {
    100: "مِيَّة",     200: "مِيَّتين",  300: "تُلتُمِيَّة",
    400: "رُبعُمِيَّة", 500: "خُمسُمِيَّة", 600: "سِتُّمِيَّة",
    700: "سُبعُمِيَّة", 800: "تُمنُمِيَّة", 900: "تُسعُمِيَّة",
}

THOUSANDS: Dict[int, str] = {
    1000: "أَلف",      2000: "أَلفين",      3000: "تَلات آلاف",
    4000: "أَربَع آلاف", 5000: "خَمس آلاف", 6000: "سِتّ آلاف",
    7000: "سَبع آلاف", 8000: "تَمَن آلاف", 9000: "تِسع آلاف",
}


# ==========================================
# 1. EGYPTIAN PHONOLOGY
# ==========================================

# Words where ث → ت in Egyptian colloquial
_THAA_TO_TAA = {
    "ثلاثة": "تلاتة", "ثمانية": "تمانية", "ثلاثون": "تلاتين",
    "ثمانون": "تمانين", "ثلاث": "تلات", "ثمان": "تمان",
    "ثلاثاء": "تلات", "الثلاثاء": "التلات",
    "ثقيل": "تقيل", "مثل": "زي", "مثلاً": "مثلاً",
    "كثير": "كتير", "حديث": "حديت", "بحث": "بحت",
    "حيث": "حيت", "ثم": "وبعدين", "ثانية": "تانية",
    "ثانٍ": "تاني", "الثاني": "التاني", "ثاني": "تاني",
}

# Words where ذ → د in Egyptian colloquial
_DHAAL_TO_DAAL = {
    "ذلك": "ده", "ذهب": "راح", "يذهب": "يروح",
    "ذكر": "اتذكر", "تذكر": "افتكر",
    "أخذ": "خد", "يأخذ": "ياخد",
    "هكذا": "كده", "لذلك": "عشان كده",
    "إذن": "يعني", "إذاً": "يعني",
    "منذ": "من", "إذا": "لو",
}

# Words where ظ → ض in Egyptian colloquial
_DHAA_TO_DAAD = {
    "بالضبط": "بالظبط",   # Already Egyptian
    "محظور": "ممنوع",
    "ظهر": "ضهر",
    "ظروف": "ضروف",
    "ظريف": "ضريف",
    "حظ": "حض",
    "انتظر": "استنى",
}

_PHONOLOGY_RE = re.compile(
    r'\b(' + '|'.join(re.escape(w) for w in
                      list(_THAA_TO_TAA) + list(_DHAAL_TO_DAAL) + list(_DHAA_TO_DAAD)
                      ) + r')\b',
    re.UNICODE
)

_PHONOLOGY_MAP = {**_THAA_TO_TAA, **_DHAAL_TO_DAAL, **_DHAA_TO_DAAD}


def apply_egyptian_phonology(text: str) -> str:
    """
    Apply Egyptian phonology: convert formal ث/ذ/ظ sounds to their
    colloquial Egyptian equivalents where appropriate.
    """
    if not text:
        return text
    return _PHONOLOGY_RE.sub(lambda m: _PHONOLOGY_MAP.get(m.group(0), m.group(0)), text)


# ==========================================
# 2. TACTICAL TASHKEEL
# ==========================================

_TASHKEEL_RE = re.compile(r'[\u064B-\u065F\u0670\u0640]')

def strip_excessive_tashkeel(text: str) -> str:
    """
    Remove tashkeel EXCEPT on words we intentionally placed it on.
    Uses word-boundary tokenization to avoid stripping protected words.
    """
    if not text:
        return text

    protected = set(JEEM_WORDS.values()) | set(PRONUNCIATION_FIXES.values())

    # Tokenize keeping punctuation attached so protected words match exactly
    tokens = re.split(r'(\s+)', text)
    result = []
    for token in tokens:
        if token in protected:
            result.append(token)
        else:
            result.append(_TASHKEEL_RE.sub('', token))
    return ''.join(result)


def apply_tactical_tashkeel(text: str) -> str:
    """
    Apply strategic tashkeel ONLY on words that need it for correct
    TTS pronunciation. Processes longest words first to avoid partial matches.
    """
    if not text:
        return text

    # Sort by length descending so "دُكتورة" is tried before "دُكتور"
    sorted_jeem = sorted(JEEM_WORDS.items(), key=lambda kv: len(kv[0]), reverse=True)
    sorted_fixes = sorted(PRONUNCIATION_FIXES.items(), key=lambda kv: len(kv[0]), reverse=True)

    for word, fixed in sorted_jeem:
        text = re.sub(r'(?<![\u0600-\u06FF])' + re.escape(word) + r'(?![\u0600-\u06FF])', fixed, text)

    for word, fixed in sorted_fixes:
        text = re.sub(r'(?<![\u0600-\u06FF])' + re.escape(word) + r'(?![\u0600-\u06FF])', fixed, text)

    return text


# ==========================================
# 3. FIX TA MARBUTA: ه → ة
# ==========================================

_HAA_EXCEPTIONS = frozenset([
    "فيه", "عليه", "منه", "إليه", "به", "له", "معاه", "وراه",
    "قدامه", "جنبه", "تحته", "فوقه", "بينه", "عنده", "أنه", "إنه",
    "لأنه", "كأنه", "هو", "هي", "أه", "ايه", "إيه", "ده",
    "ليه", "بتاعه", "نفسه", "وجهه", "شبهه", "اتجاه", "تنبيه",
    "توجيه", "تشابه", "تمويه", "فقه", "وجه", "سفه", "عبده", "نزيه",
    "مكروه", "مشبوه", "موجه", "منبه", "شبه", "عمله", "كله",
    "ازاي", "هوه", "هيه", "معاه", "وراه", "عنده", "عندها", "عندهم",
])

_TA_MARBUTA_PATTERN = re.compile(
    r'\b([\u0621-\u063A\u0641-\u064A][\u0600-\u06FF]*?)(\u0647)(?=\s|[.،؟!:\-]|$)',
    re.UNICODE
)


def fix_ta_marbuta(text: str) -> str:
    """DISABLED: Removing this because it aggressively mangles names ending in ه (e.g. عبدالله -> عبداللة, طه -> طة)."""
    return text


# ==========================================
# 4. FIX ALEF MAQSURA / YA CONFUSION
# ==========================================

_ALEF_MAQSURA_WORDS = frozenset([
    "على", "إلى", "حتى", "مستشفى", "مبنى", "معنى", "مجرى",
    "مدى", "أخرى", "كبرى", "صغرى", "أولى", "منى", "مصطفى",
    "هدى", "سلمى", "ليلى", "موسى", "عيسى", "يحيى", "مرتضى",
    "مجتبى", "مرمى", "مقهى",
])

def fix_alef_maqsura(text: str) -> str:
    """Convert terminal ى to ي for all words NOT in the known Alef Maqsura list. This fixes names like مجدى"""
    if not text:
        return text
    
    def replacer(match):
        word = match.group(0)
        if word in _ALEF_MAQSURA_WORDS:
            return word
        return word[:-1] + 'ي'

    return re.sub(r'\b[\u0600-\u06FF]+ى\b', replacer, text)


# ==========================================
# 5. SMART LIST PUNCTUATION
# ==========================================

def is_list_context(text: str) -> bool:
    """Detect if text contains a numbered list structure."""
    return bool(re.search(r'\b\d{1,2}\s*[-.)]\s*[\u0600-\u06FF]', text))


def add_list_punctuation(text: str) -> str:
    """Add punctuation for numbered lists."""
    if not text:
        return text
    text = re.sub(r'\b(\d{1,2})\s+([\u0600-\u06FF])', r'\1، \2', text)
    text = re.sub(r'([\u0600-\u06FF])\s+(\d{1,2})\b', r'\1، \2 ', text)
    return text


def ensure_punctuation(text: str) -> str:
    """Add punctuation for natural TTS pauses, preserving newlines."""
    if not text:
        return text

    # Process line by line to preserve newlines
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        if not line.strip():
            processed_lines.append(line)
            continue
            
        if is_list_context(line):
            line = add_list_punctuation(line)

        punct_chars = set('.,،؟?!؛:')
        punct_count = sum(1 for c in line if c in punct_chars)
        word_count = len(line.split())

        if word_count <= 5 or (punct_count > 0 and word_count / max(punct_count, 1) < 12):
            line = line.rstrip()
            if line and line[-1] not in punct_chars:
                line += '.'
            processed_lines.append(line)
            continue

        if not is_list_context(line):
            words = line.split(' ') # Split exactly on spaces to preserve any internal spacing
            result = []
            since_last = 0

            for i, word in enumerate(words):
                if not word: # Handle multiple spaces
                    result.append(word)
                    continue
                    
                result.append(word)
                since_last += 1

                if word and word[-1] in punct_chars:
                    since_last = 0
                    continue

                # Pause every 10 words (was 8 — gives more natural flow)
                if since_last >= 10 and i < len(words) - 1:
                    result[-1] = word + '،'
                    since_last = 0

            line = ' '.join(result)

        line = re.sub(r'،+', '،', line)
        line = re.sub(r'،\s*،', '،', line)

        line = line.rstrip()
        if line and line[-1] not in punct_chars:
            line += '.'
            
        processed_lines.append(line)

    return '\n'.join(processed_lines)


# ==========================================
# 6. NUMBER CONVERSION
# ==========================================

def number_to_arabic_words(num: int) -> str:
    """Convert number to Egyptian Arabic words with tactical tashkeel."""
    if num < 0:
        return str(num)
    if num == 0:
        return ONES[0]

    parts = []

    if num >= 1000:
        thousands = (num // 1000) * 1000
        parts.append(THOUSANDS.get(thousands, str(thousands)))
        num = num % 1000

    if num >= 100:
        hundreds = (num // 100) * 100
        if hundreds in HUNDREDS:
            parts.append(HUNDREDS[hundreds])
        num = num % 100

    if num > 0:
        if num in ONES:
            parts.append(ONES[num])
        elif num in TENS:
            parts.append(TENS[num])
        else:
            tens = (num // 10) * 10
            ones = num % 10
            parts.append(f"{ONES[ones]} و {TENS[tens]}")

    return " و ".join(parts)


def normalize_numbers_in_text(text: str, keep_list_numbers: bool = True) -> str:
    """Convert numbers to words, optionally keeping list numbers as digits."""
    def replace_number(match):
        num_str = match.group(0)
        num = int(num_str)
        if keep_list_numbers and is_list_context(text) and num < 100:
            return num_str
        return number_to_arabic_words(num)

    # Expand 4-digit years naturally: 2024 → "20 24" → handled below
    text = re.sub(r'\b(19|20)(\d{2})\b', r'\1 \2', text)

    # Common years
    year_map = {
        "20 26": "أَلفين وسِتَّة وعِشرين",
        "20 25": "أَلفين وخَمسَة وعِشرين",
        "20 24": "أَلفين وأَربَعة وعِشرين",
        "20 23": "أَلفين وتَلاتَة وعِشرين",
        "19 90": "تِسعُمِيَّة وتِسعين",
        "19 80": "تِسعُمِيَّة وتَمانين",
        "19 70": "تِسعُمِيَّة وسَبعين",
    }
    for k, v in year_map.items():
        text = text.replace(k, v)

    text = re.sub(r'\b\d+\b', replace_number, text)
    return text


# ==========================================
# 7. TIME NORMALIZATION
# ==========================================

def normalize_time_format(text: str) -> str:
    """Normalize time with Egyptian Arabic words for clear pronunciation."""

    def get_hour_word(h: int) -> str:
        hour_words = {
            1: "واحدة", 2: "اتنين", 3: "تلاتة", 4: "أربعة",
            5: "خمسة",  6: "ستة",   7: "سبعة",  8: "تمانية",
            9: "تسعة",  10: "عشرة", 11: "حداشر", 12: "اتناشر",
        }
        return hour_words.get(h % 12 or 12, ONES.get(h, str(h)))

    def replace_time(match):
        hour   = int(match.group(2))
        minute = int(match.group(3)) if match.group(3) else 0
        period_marker = match.group(4) or ""
        m_lower = period_marker.lower()

        period = ""
        if any(x in m_lower for x in ["ص", "am", "صباح"]):
            period = "صَباحًا"
        elif any(x in m_lower for x in ["م", "pm", "مساء", "ليل"]):
            period = "مَساءً"

        hour_word = get_hour_word(hour)
        minute_part = ""

        if minute == 0:
            minute_part = ""
        elif minute == 30:
            minute_part = "و نُص"
        elif minute == 15:
            minute_part = "و رُبع"
        elif minute == 20:
            minute_part = "و تِلت"
        elif minute == 40:
            hour_word   = get_hour_word(hour + 1)
            minute_part = "إلَّا تِلت"
        elif minute == 45:
            hour_word   = get_hour_word(hour + 1)
            minute_part = "إلَّا رُبع"
        elif minute == 50:
            hour_word   = get_hour_word(hour + 1)
            minute_part = "إلَّا عَشَرة"
        elif minute == 55:
            hour_word   = get_hour_word(hour + 1)
            minute_part = "إلَّا خَمسَة"
        elif minute == 10:
            minute_part = "و عَشَرة"
        elif minute == 5:
            minute_part = "و خَمسَة"
        else:
            minute_part = f"و {number_to_arabic_words(minute)} دَقيقة"

        parts = [p for p in [hour_word, minute_part, period] if p]
        return " ".join(parts).strip()

    suffix = r"(صباحًا|مساءً|صباحا|مساء|AM|PM|am|pm|ص|م)"
    return re.sub(
        r'(الساعة\s*)?(\d{1,2})(?:[:.](\d{2}))?\s*' + suffix + r'?',
        replace_time,
        text
    )


# ==========================================
# 8. CURRENCY & DATES
# ==========================================

def normalize_currency(text: str) -> str:
    """Normalize currency to Egyptian Arabic words."""
    def replace_money(match):
        num   = int(match.group(1))
        words = number_to_arabic_words(num)
        return f"{words} جُنَيه"

    text = re.sub(
        r'\b(\d+)\s*(جنيه|جنيهات|EGP|LE|egp|le)\b',
        replace_money, text, flags=re.IGNORECASE
    )
    return text


def normalize_dates(text: str) -> str:
    """Normalize weekdays to Egyptian colloquial."""
    weekdays = {
        "السبت":    "السَّبت",
        "الأحد":    "الأَحَد",
        "الاثنين":  "الإتنِين",
        "الإثنين":  "الإتنِين",
        "الثلاثاء": "التَّلات",
        "الأربعاء": "الأَربَع",
        "الخميس":   "الخَميس",
        "الجمعة":   "الجُمعَة",
    }
    for formal, colloquial in weekdays.items():
        text = text.replace(formal, colloquial)
    return text


# ==========================================
# 9. MAIN PIPELINE
# ==========================================

def normalize_arabic_for_tts(text: str, keep_list_numbers: bool = True) -> str:
    """
    Ultimate TTS normalization pipeline:
      MSA → Egyptian Ammya → phonology → tashkeel → numbers → time → punctuation.
    """
    if not text:
        return text

    # ── Step 1: MSA → Egyptian dialect (THE CRITICAL STEP — was missing before)
    from .egyptian_mappings import apply_egyptian_mappings
    text = apply_egyptian_mappings(text)

    # ── Step 2: Egyptian phonology (ث→ت, ذ→د, ظ→ض in common words)
    text = apply_egyptian_phonology(text)

    # ── Step 3: Strip random LLM tashkeel (must come after dialect mapping)
    text = strip_excessive_tashkeel(text)

    # ── Step 4: Fix Ta Marbuta
    text = fix_ta_marbuta(text)

    # ── Step 5: Fix Alef Maqsura
    text = fix_alef_maqsura(text)

    # ── Step 6: Normalize weekday names
    text = normalize_dates(text)

    # ── Step 7: Normalize time formats
    text = normalize_time_format(text)

    # ── Step 8: Punctuation character normalization
    text = text.replace('?', '؟')
    text = text.replace(',', '،')

    # ── Step 9: Currency
    text = normalize_currency(text)

    # ── Step 10: Ensure natural pause punctuation
    text = ensure_punctuation(text)

    # ── Step 11: Numbers → words
    text = normalize_numbers_in_text(text, keep_list_numbers=keep_list_numbers)

    # ── Step 12: Apply tactical tashkeel for correct TTS pronunciation
    text = apply_tactical_tashkeel(text)

    # ── Step 13: Final cleanup
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*',     r'\1', text)
    text = re.sub(r'\.{2,}',          '.',   text)
    text = re.sub(r',{2,}',           '،',   text)
    text = re.sub(r'  +',             ' ',   text)

    # ── Step 14: STRIP ALL TASHKEEL
    # OpenAI tts-1 treats heavily voweled Egyptian slang as Farsi/Urdu.
    # Stripping all diacritics forces it to use its Arabic phonetic base.
    text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)

    return text.strip()


# ==========================================
# 10. MEDICAL CONTEXT SHORTCUT
# ==========================================

def normalize_medical_text(text: str) -> str:
    """
    Medical text normalization — applies full pipeline.
    The egyptian_mappings layer already handles most medical substitutions;
    this function exists as a named convenience entry-point.
    """
    return normalize_arabic_for_tts(text)