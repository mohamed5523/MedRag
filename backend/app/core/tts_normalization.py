"""
ULTIMATE Egyptian Arabic TTS Normalization
- Smart tashkeel placement for pronunciation consistency
- Proper ج (jeem) pronunciation hints
- List punctuation for natural pauses
- Egyptian dialect optimization
"""

import re
from typing import Dict, Set

# ==========================================
# PRONUNCIATION DICTIONARIES
# ==========================================

# Words with ج that should be pronounced as "J" (soft jeem)
# These need fatha on the jeem to ensure correct pronunciation
JEEM_NAMES = {
    "جورج": "Jeorge",
    "جورجينا": "Jeorgina",
    "جورجي": "Jeorgy",
    "جورجى": "Jeorgy",
    "جيم": "Jim",
    "جيمس": "James",
    "جين": "Jean",
    "جينا": "Gina",
    "جاك": "Jack",
    "جاكلين": "Jacqueline",
    "جاكسون": "Jackson",
    "جان": "Jan",
    "جانيت": "Janet",
    "جوزيف": "Joseph",
    "جوليا": "Julia",
    "جوليان": "Julian",
    "جوني": "Johnny",
    "جيرالد": "Gerald",
    "جيسيكا": "Jessica",
    "جوناثان": "Jonathan",
    "جنيفر": "Jennifer",
    "جيسي": "Jesse",
    "جوردان": "Jordan",
}

JEEM_WORDS = JEEM_NAMES

# Common ambiguous words that need tashkeel for correct pronunciation
PRONUNCIATION_FIXES = {
    # Medical terms
    "دكتور": "دُكتور",
    "دكتورة": "دُكتورة",
    "مستشفى": "مُستشفى",
    "عيادة": "عِيادة",
    "صيدلية": "صَيدَلِية",
    "كشف": "كَشف",
    "تحليل": "تَحليل",
    "تحاليل": "تَحاليل",
    "اشعة": "أَشِعَّة",
    "أشعة": "أَشِعَّة",
    "حقنة": "حُقنة",
    "علاج": "عِلاج",
    "وصفة": "وَصفة",
    "عملية": "عَمَلِية",
    "ميعاد": "مَعاد",
    "معاد": "مَعاد",
    "موعد": "مَعاد",
    
    # Common verbs (present tense markers)
    "يروح": "يِروح",
    "تروح": "تِروح",
    "نروح": "نِروح",
    "يقدر": "يِقدر",
    "تقدر": "تِقدر",
    "نقدر": "نِقدر",
    "يعمل": "يِعمل",
    "تعمل": "تِعمل",
    "نعمل": "نِعمل",
    "يشوف": "يِشوف",
    "تشوف": "تِشوف",
    "نشوف": "نِشوف",
    
    # Common words with ambiguity
    "عند": "عِند",
    "فوق": "فُوق",
    "تحت": "تَحت",
    "بين": "بِين",
    "معاه": "مَعاه",
    "ليه": "لِيه",
    "ازاي": "إزَّاي",
    "امتى": "إمتَى",
    "فين": "فِين",
    "مين": "مِين",
    
    # Time words
    "الصبح": "صَباحًا",
    "صباحا": "صَباحًا",
    "صباحًا": "صَباحًا",
    "الضهر": "الضُهر",
    "العصر": "العَصر",
    "المغرب": "المَغرِب",
    "بالليل": "مَساءً",
    "مساء": "مَساءً",
    "مساءً": "مَساءً",
    
    # Common Egyptian words
    "دلوقتي": "دِلوَقتي",
    "النهاردة": "النَّهارْدَه",
    "بكرة": "بُكرَه",
    "امبارح": "إمبارِح",
    "ماشي": "ماشِي",
    "تمام": "تَمام",
    "خلاص": "خَلاص",
    "طيب": "طَيِّب",
    
    # Numbers (when spelled out)
    "واحد": "واحِد",
    "اتنين": "إتنِين",
    "تلاتة": "تَلاتَة",
    "اربعة": "أَربَعة",
    "خمسة": "خَمسَة",
    "ستة": "سِتَّة",
    "سبعة": "سَبعَة",
    "تمانية": "تَمانيَة",
    "تسعة": "تِسعَة",
    "عشرة": "عَشَرة",
    
    # Names
    "زكى": "زكي",
}

# ==========================================
# NUMBER DEFINITIONS - EGYPTIAN STYLE

# ==========================================

ONES = {
    0: "صفر", 1: "واحِد", 2: "إتنِين", 3: "تَلاتَة", 4: "أَربَعة",
    5: "خَمسَة", 6: "سِتَّة", 7: "سَبعَة", 8: "تَمانيَة", 9: "تِسعَة",
    10: "عَشَرة", 11: "حِداشَر", 12: "إتناشَر", 13: "تَلاتاشَر",
    14: "أَربَعتاشَر", 15: "خَمَستاشَر", 16: "سِتَّاشَر", 17: "سَبَعتاشَر",
    18: "تَمَنتاشَر", 19: "تِسَعتاشَر"
}

TENS = {
    20: "عِشرين", 30: "تَلاتين", 40: "أَربَعين", 50: "خَمسين",
    60: "سِتِّين", 70: "سَبعين", 80: "تَمانين", 90: "تِسعين"
}

HUNDREDS = {
    100: "مِيَّة", 200: "مِيَّتين", 300: "تُلتُمِيَّة", 400: "رُبعُمِيَّة",
    500: "خُمسُمِيَّة", 600: "سِتُّمِيَّة", 700: "سُبعُمِيَّة", 800: "تُمنُمِيَّة", 900: "تُسعُمِيَّة"
}

THOUSANDS = {
    1000: "أَلف", 2000: "أَلفين", 3000: "تَلات آلاف", 4000: "أَربَع آلاف",
    5000: "خَمس آلاف", 6000: "سِتّ آلاف", 7000: "سَبع آلاف", 8000: "تَمَن آلاف", 9000: "تِسع آلاف"
}


# ==========================================
# 1. TACTICAL TASHKEEL APPLICATION
# ==========================================

def apply_tactical_tashkeel(text: str) -> str:
    """
    Apply strategic tashkeel ONLY on words that need it for correct pronunciation.
    This is different from full tashkeel - we only mark critical pronunciation points.
    """
    if not text:
        return text
    
    # First, apply Jeem pronunciation fixes (highest priority)
    for word, fixed in JEEM_WORDS.items():
        # Use word boundaries to avoid partial matches
        text = re.sub(r'\b' + re.escape(word) + r'\b', fixed, text, flags=re.IGNORECASE)
    
    # Then apply other pronunciation fixes
    for word, fixed in PRONUNCIATION_FIXES.items():
        text = re.sub(r'\b' + re.escape(word) + r'\b', fixed, text, flags=re.IGNORECASE)
    
    return text


def strip_excessive_tashkeel(text: str) -> str:
    """
    Remove tashkeel EXCEPT on words we intentionally marked.
    This keeps our tactical tashkeel while removing LLM-generated random tashkeel.
    """
    if not text:
        return text
    
    # Get all our intentionally marked words
    protected_words = set(JEEM_WORDS.values()) | set(PRONUNCIATION_FIXES.values())
    
    words = text.split()
    result = []
    
    _TASHKEEL_RE = re.compile(r'[\u064B-\u065F\u0670\u0640]')
    
    for word in words:
        # Check if this word is in our protected list (with tashkeel)
        if word in protected_words:
            # Keep as is
            result.append(word)
        else:
            # Strip tashkeel from this word
            cleaned = _TASHKEEL_RE.sub('', word)
            result.append(cleaned)
    
    return ' '.join(result)


# ==========================================
# 2. FIX TA MARBUTA: ه → ة
# ==========================================

_HAA_EXCEPTIONS = frozenset([
    "فيه", "عليه", "منه", "إليه", "به", "له", "معاه", "وراه",
    "قدامه", "جنبه", "تحته", "فوقه", "بينه", "عنده", "أنه", "إنه",
    "لأنه", "كأنه", "هو", "هي", "أه", "ايه", "إيه", "ده",
    "ليه", "بتاعه", "نفسه", "وجهه", "شبهه", "اتجاه", "تنبيه",
    "توجيه", "تشابه", "تمويه", "فقه", "وجه", "سفه", "عبده", "نزيه",
    "مكروه", "مشبوه", "موجه", "منبه", "شبه", "عمله", "كله", "ازيه",
    "امته", "ازاي",
])

def fix_ta_marbuta(text: str) -> str:
    """Replace word-final ه with ة, skipping known exceptions."""
    if not text:
        return text

    _ar = r'[\u0621-\u063A\u0641-\u064A]'
    _pattern = re.compile(
        r'\b(' + _ar + r'[\u0600-\u06FF]*?)(\u0647)(?=\s|[.،؟!:\-]|$)',
        re.UNICODE
    )

    def _replace(m: re.Match) -> str:
        prefix = m.group(1)
        full_word = prefix + 'ه'
        if full_word in _HAA_EXCEPTIONS:
            return m.group(0)
        return prefix + 'ة'

    return _pattern.sub(_replace, text)


# ==========================================
# 3. FIX ALEF MAQSURA / YA CONFUSION
# ==========================================

_ALEF_MAQSURA_WORDS = frozenset([
    "على", "إلى", "حتى", "مستشفى", "مبنى", "معنى", "مجرى",
    "مدى", "أخرى", "كبرى", "صغرى", "أولى", "منى", "مصطفى",
    "هدى", "سلمى", "ليلى", "موسى", "عيسى", "يحيى", "مرتضى",
])

def fix_alef_maqsura(text: str) -> str:
    """Fix common ي→ى mistakes."""
    if not text:
        return text
    for word in _ALEF_MAQSURA_WORDS:
        wrong = word[:-1] + 'ي'
        text = re.sub(r'\b' + re.escape(wrong) + r'\b', word, text)
    return text


# ==========================================
# 4. SMART LIST PUNCTUATION
# ==========================================

def is_list_context(text: str) -> bool:
    """Detect if text contains a numbered list structure."""
    pattern = r'\b\d\s+[^\d]+\s+\d\b'
    return bool(re.search(pattern, text))


def add_list_punctuation(text: str) -> str:
    """Add punctuation for numbered lists."""
    if not text:
        return text
    
    # Add comma after numbers: "1 name" → "1، name"
    text = re.sub(r'\b(\d{1,2})\s+([ا-ي])', r'\1، \2', text)
    
    # Add comma before next number: "name 2" → "name، 2"
    text = re.sub(r'([ا-ي])\s+(\d{1,2})\s', r'\1، \2 ', text)
    
    return text


def ensure_punctuation(text: str) -> str:
    """Add punctuation for natural TTS pauses."""
    if not text:
        return text

    # Check if this is a list context
    if is_list_context(text):
        text = add_list_punctuation(text)

    punct_chars = set('.,،؟?!؛:')
    punct_count = sum(1 for c in text if c in punct_chars)
    word_count = len(text.split())

    if word_count <= 5 or (punct_count > 0 and word_count / max(punct_count, 1) < 12):
        text = text.rstrip()
        if text and text[-1] not in punct_chars:
            text += '.'
        return text

    if not is_list_context(text):
        words = text.split()
        result = []
        since_last = 0

        for i, word in enumerate(words):
            result.append(word)
            since_last += 1
            
            if word and word[-1] in punct_chars:
                since_last = 0
                continue
            
            if since_last >= 8 and i < len(words) - 1:
                result[-1] = word + '،'
                since_last = 0

        text = ' '.join(result)

    text = re.sub(r'،+', '،', text)
    text = re.sub(r'،\s*،', '،', text)
    
    text = text.rstrip()
    if text and text[-1] not in punct_chars:
        text += '.'
    
    return text


# ==========================================
# 5. NUMBER CONVERSION
# ==========================================

def number_to_arabic_words(num: int) -> str:
    """Convert number to Egyptian Arabic words with tactical tashkeel."""
    if num < 0: return str(num)
    if num == 0: return ONES[0]

    parts = []
    
    if num >= 1000:
        thousands = (num // 1000) * 1000
        if thousands in THOUSANDS:
            parts.append(THOUSANDS[thousands])
        else:
            parts.append(str(thousands))
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

    # Handle years
    text = re.sub(r'\b(19|20)(\d{2})\b', r'\1 \2', text)
    
    text = text.replace("20 26", "أَلفين وسِتَّة وعِشرين")
    text = text.replace("20 25", "أَلفين وخَمسَة وعِشرين")
    text = text.replace("20 24", "أَلفين وأَربَعة وعِشرين")
    
    text = re.sub(r'\b\d+\b', replace_number, text)
    
    return text


# ==========================================
# 6. TIME NORMALIZATION
# ==========================================

def normalize_time_format(text: str) -> str:
    """Normalize time with tactical tashkeel for clear pronunciation."""
    
    def replace_time(match):
        hour = int(match.group(2))
        minute = int(match.group(3))
        period_marker = match.group(4) or ""

        period = ""
        m_lower = period_marker.lower()
        if any(x in m_lower for x in ["ص", "am", "صباح"]):
            period = "صَباحًا"
        elif any(x in m_lower for x in ["م", "pm", "مساء", "ليل"]):
            period = "مَساءً"

        def get_hour_word(h):
            if h == 1:
                return "واحده"
            elif h == 2:
                return "اتنين"
            elif h == 3:
                return "تلاته"
            elif h == 4:
                return "اربعه"
            elif h == 5:
                return "خمسه"
            elif h == 6:
                return "سته"
            elif h == 7:
                return "سبعه"
            elif h == 8:
                return "تمانيه"
            elif h == 9:
                return "تسعه"
            elif h == 10:
                return "عشره"
            elif h == 11:
                return "حداشر"
            elif h == 12:
                return "اتناشر"
            else:
                return ONES.get(h, str(h))

        hour_word = get_hour_word(hour)

        minute_part = ""
        if minute == 0:
            minute_part = ""
        elif minute == 30:
            minute_part = "و نُصّ"
        elif minute == 15:
            minute_part = "و رُبع"
        elif minute == 20:
            minute_part = "و تِلت"
        elif minute == 40:
            next_hour = hour + 1 if hour < 12 else 1
            hour_word = get_hour_word(next_hour)
            minute_part = "إلّا تِلت"
        elif minute == 45:
            next_hour = hour + 1 if hour < 12 else 1
            hour_word = get_hour_word(next_hour)
            minute_part = "إلّا رُبع"
        elif minute == 50:
            next_hour = hour + 1 if hour < 12 else 1
            hour_word = get_hour_word(next_hour)
            minute_part = "إلّا عَشَرة"
        elif minute == 55:
            next_hour = hour + 1 if hour < 12 else 1
            hour_word = get_hour_word(next_hour)
            minute_part = "إلّا خَمسَة"
        elif minute == 10:
            minute_part = "و عَشَرة"
        elif minute == 5:
            minute_part = "و خَمسَة"
        else:
            minute_part = f"و {number_to_arabic_words(minute)} دَقيقة"

        parts = []
        if minute == 0:
            parts = [hour_word, period]
        else:
            parts = [hour_word, minute_part, period]
        
        result = " ".join([p for p in parts if p]).strip()
        return result

    suffix_pattern = r"(صباحًا|مساءً|صباحا|مساء|AM|PM|am|pm|ص|م)"
    return re.sub(
        r'(الساعة\s*)?(\d{1,2})[:.](\d{2})\s*' + suffix_pattern + r'?',
        replace_time,
        text
    )


# ==========================================
# 7. CURRENCY & DATES
# ==========================================

def normalize_currency(text: str) -> str:
    """Normalize currency with tashkeel."""
    def replace_money(match):
        num = int(match.group(1))
        words = number_to_arabic_words(num)
        return f"{words} جُنَيه"
    
    text = re.sub(
        r'\b(\d+)\s*(جنيه|جنيهات|EGP|LE|egp|le)\b',
        replace_money,
        text,
        flags=re.IGNORECASE
    )
    return text


def normalize_dates(text: str) -> str:
    """Normalize dates to Egyptian with tashkeel."""
    weekdays = {
        "السبت": "السَّبت",
        "الأحد": "الأَحَد",
        "الاثنين": "الإتنِين",
        "الإثنين": "الإتنِين",
        "الثلاثاء": "التَّلات",
        "الأربعاء": "الأَربَع",
        "الخميس": "الخَميس",
        "الجمعة": "الجُمعَة",
    }
    
    for formal, colloquial in weekdays.items():
        text = text.replace(formal, colloquial)
    
    return text


# ==========================================
# 8. MAIN PIPELINE
# ==========================================

def normalize_arabic_for_tts(text: str, keep_list_numbers: bool = True) -> str:
    """
    Ultimate TTS normalization with tactical tashkeel for pronunciation consistency.
    
    Args:
        keep_list_numbers: Keep list numbers as digits instead of converting to words
    """
    if not text:
        return text

    # 1. Strip any existing random tashkeel from LLM output
    text = strip_excessive_tashkeel(text)

    # 2. Fix Ta Marbuta
    text = fix_ta_marbuta(text)

    # 3. Fix Alef Maqsura
    text = fix_alef_maqsura(text)

    # 4. Normalize dates
    text = normalize_dates(text)

    # 5. Normalize time
    text = normalize_time_format(text)

    # 6. Fix punctuation
    text = text.replace('?', '؟')
    text = text.replace(',', '،')

    # 7. Currency
    text = normalize_currency(text)

    # 8. Punctuation (before number conversion)
    text = ensure_punctuation(text)

    # 9. Numbers
    text = normalize_numbers_in_text(text, keep_list_numbers=keep_list_numbers)

    # 10. Apply tactical tashkeel for pronunciation (CRITICAL STEP)
    text = apply_tactical_tashkeel(text)

    # 11. Clean formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r',{2,}', '،', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ==========================================
# 9. MEDICAL CONTEXT
# ==========================================

def normalize_medical_text(text: str) -> str:
    """Medical text normalization with pronunciation fixes."""
    medical_replacements = {
        "موعد": "مَعاد",
        "مواعيد": "مَعاديد",
        "ميعاد": "مَعاد",
        "فحص": "كَشف",
        "اختبار": "تَحليل",
        "اختبارات": "تَحاليل",
        "تصوير": "أَشِعَّة",
        "إبرة": "حُقنة",
        "روشتة": "وَصفة",
        "جراحة": "عَمَلِية",
        "وصفة طبية": "روشتة",
        "طبيب": "دُكتور",
        "طبيبة": "دُكتورة",
        "أخصائي": "أَخُصّائي",
        "استشاري": "إستِشاري",
        "تخدير": "بنج",
        "غرفة العمليات": "أوضة العمليات",
        "طوارئ": "طَوارئ",
        "حضانة": "حَضانة",
        "عناية مركزة": "عِناية مُرَكَّزة",
        "ضغط": "ضَغط",
        "سكر": "سُكَّر",
        "حرارة": "حَرارة",
        "نبض": "نَبض",
        "حساسية": "حَساسِيَّة",
        "التهاب": "إلتِهاب",
        "مضاد حيوي": "مُضاد حَيَوي",
        "مسكن": "مُسَكِّن",
    }
    
    for formal, colloquial in medical_replacements.items():
        text = re.sub(r'\b' + formal + r'\b', colloquial, text)
    
    return normalize_arabic_for_tts(text)