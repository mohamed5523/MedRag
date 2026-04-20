"""
Radiology Knowledge Base
========================
Provides prep instructions, machine guidance, and contraindication info
for all radiology modalities. Used as MCP fallback context when no
pricing / schedule data exists in the clinic system.
"""

from __future__ import annotations

import re
from typing import Optional

# ── per-modality structured knowledge ────────────────────────────────────────

RADIOLOGY_MODALITIES: dict[str, dict] = {
    "mri": {
        "arabic_name": "الرنين المغناطيسي (MRI)",
        "prep": (
            "عادةً مش محتاج صيام. "
            "لو بالصبغة: صيام 4 ساعات + تحليل كرياتينين (وظائف كلى) قبلها."
        ),
        "machine_type": (
            "الجهاز في الغالب مقفول (Closed Bore). "
            "لو عندك رهاب أماكن ضيقة أو وزنك تقيل، اتصل بالمستشفى للاستفسار عن توافر جهاز مفتوح."
        ),
        "contraindications": (
            "ممنوع تماماً لو عندك Pacemaker أو Cochlear Implant. "
            "معظم الشرائح والمسامير من التيتانيوم الحديثة آمنة تماماً، "
            "بس لازم تبلغ الفني بنوع الشريحة وتاريخ التركيب."
        ),
        "duration": "من 30 دقيقة لساعة حسب المنطقة اللي بتتصور.",
        "sub_clinic": "اشعه",
    },
    "ct": {
        "arabic_name": "الأشعة المقطعية (CT / سكانر)",
        "prep": (
            "بدون صبغة: مش محتاج صيام. "
            "بالصبغة: صيام 4 ساعات + تحليل كرياتينين. "
            "لو بتاخد ميتفورمين (Metformin): وقفها 48 ساعة بعد الصبغة باستشارة دكتورك."
        ),
        "machine_type": "الجهاز مفتوح دايماً، مش زي الرنين.",
        "contraindications": (
            "الحوامل: يفضل تُتجنب لوجود إشعاع — السونار بديل أأمن. "
            "حساسية من اليود: ممنوع الصبغة."
        ),
        "duration": "سريعة جداً، من 5 إلى 15 دقيقة. النتيجة في نفس اليوم عادةً.",
        "sub_clinic": "اشعه مقطعيه",
    },
    "ultrasound": {
        "arabic_name": "الأشعة التلفزيونية (سونار / Ultrasound)",
        "prep": (
            "الكبد والمرارة والبنكرياس: صيام 6-8 ساعات. "
            "الحوض والمثانة والبروستاتا: اشرب كوباية مية كاملة ولا تتبول. "
            "الجنين والحمل: مش محتاج صيام."
        ),
        "machine_type": "الجهاز مفتوح تماماً ومريح.",
        "contraindications": "آمن تماماً، مفيش أي إشعاع. مناسب للحوامل والأطفال.",
        "duration": "من 15 إلى 30 دقيقة.",
        "sub_clinic": "اشعه تلفزيونيه",
    },
    "doppler": {
        "arabic_name": "أشعة الدوبلر (Doppler)",
        "prep": "عادةً مش محتاج صيام. للدوالي والأوردة: ممكن الدكتور يطلبك تقف أثناء الفحص.",
        "machine_type": "جهاز سونار مفتوح تماماً.",
        "contraindications": "آمن تماماً، مفيش إشعاع.",
        "duration": "من 20 إلى 40 دقيقة.",
        "sub_clinic": "اشعه تلفزيونيه",
    },
    "mammogram": {
        "arabic_name": "الماموجرام (Mammography / أشعة الثدي)",
        "prep": (
            "أفضل وقت للفحص: من اليوم 7 لـ 10 من بداية الدورة. "
            "ممنوع ديودورانت أو كريم أو طلاء أظافر يوم الفحص."
        ),
        "machine_type": "جهاز مخصص للماموجرام، مفتوح.",
        "gender_note": (
            "الفحص بيتعمل عن طريق فني أو دكتورة متخصصة. "
            "اتصل بالمستشفى مسبقاً للتأكيد من توافر دكتورة ست لو محتاج."
        ),
        "contraindications": (
            "للنساء فقط. للكشف الدوري: من سن 40+. "
            "لو في تاريخ عائلي للسرطان: يُنصح من سن 30."
        ),
        "duration": "من 10 إلى 20 دقيقة.",
        "sub_clinic": "اشعه",
    },
    "dexa": {
        "arabic_name": "ديكسا (DEXA — قياس كثافة العظام / هشاشة العظام)",
        "prep": "مش محتاج صيام. شيل المعادن والمجوهرات قبل الفحص.",
        "machine_type": "جهاز مفتوح تماماً.",
        "contraindications": "آمن جداً — إشعاع ضئيل جداً أقل بكتير من الأشعة العادية.",
        "duration": "من 10 إلى 20 دقيقة.",
        "sub_clinic": "اشعه",
    },
    "pet_scan": {
        "arabic_name": "المسح الذري (PET Scan / PET-CT)",
        "prep": (
            "صيام 6 ساعات (الماء مسموح). "
            "ممنوع تتمرن 24 ساعة قبلها. "
            "محتاج حجز مسبق قبل 24-48 ساعة على الأقل لتحضير المادة المشعة."
        ),
        "machine_type": "جهاز مشترك PET/CT، مفتوح نسبياً.",
        "contraindications": (
            "الحوامل: ممنوع تماماً. "
            "الرضاعة: تُوقف 24 ساعة بعد الفحص. "
            "مرضى السكر: لازم السكر تحت السيطرة قبل الفحص."
        ),
        "duration": "ساعة ونص لساعتين (ساعة انتظار بعد الحقنة + 30-45 دقيقة تصوير).",
        "booking_required": True,
        "sub_clinic": "اشعه مقطعيه",
    },
    "xray": {
        "arabic_name": "الأشعة العادية (X-Ray / إكس راي)",
        "prep": "مش محتاج صيام. شيل المعادن والمجوهرات من منطقة التصوير.",
        "machine_type": "جهاز مفتوح، سريع جداً.",
        "contraindications": (
            "الحوامل: بلغي الفني عشان يحمي البطن بـ Lead Apron."
        ),
        "duration": "دقائق فقط. النتيجة في نفس اليوم عادةً.",
        "same_day_result": True,
        "sub_clinic": "اشعه عاديه",
    },
    "panorama": {
        "arabic_name": "البانوراما والسيفالوميتريك (أشعة الأسنان)",
        "prep": "مش محتاج صيام. شيل الحلي والمعادن من منطقة الرأس والرقبة.",
        "machine_type": "جهاز أشعة مخصص للأسنان.",
        "routing_note": (
            "لو الطلب من دكتور أسنان: روح قسم الأسنان أول. "
            "ممكن تتعمل في قسم الأشعة كمان — اسأل الاستقبال."
        ),
        "duration": "دقيقتين فقط.",
        "sub_clinic": "أسنان",
    },
}

# ── keyword → modality detection ─────────────────────────────────────────────

_MODALITY_KEYWORDS: list[tuple[list[str], str]] = [
    (["pet scan", "بت سكان", "مسح ذري", "pet-scan", "بيتي سكان"], "pet_scan"),
    (["ماموجرام", "mammogram", "أشعة ثدي", "اشعة ثدي", "اشعه ثدي"], "mammogram"),
    (["ديكسا", "dexa", "هشاشة عظام", "كثافة عظام", "قياس عظام", "قياس الكثافة"], "dexa"),
    (["بانوراما", "باناراما", "سيفالوميتريك", "panorama", "cephalometric"], "panorama"),
    (["دوبلر", "doppler", "دوالي", "أوردة الرجل", "اوردة الرجل"], "doppler"),
    (["رنين", "مرنانة", "مرنانه", "mri", "رنين مغناطيسي"], "mri"),
    (["سكانر", "مقطعي", "مقطعية", "مقطعيه", "ct scan", "ct-scan"], "ct"),
    (["سونار", "تلفزيوني", "تلفزيونية", "تلفزيونيه", "ultrasound"], "ultrasound"),
    (["إكس راي", "x-ray", "xray", "x ray", "أشعة عادية", "اشعه عاديه", "أشعة بسيطة"], "xray"),
]

# Emergency patterns (child swallowing, severe accident needing imaging)
_EMERGENCY_PATTERNS: list[str] = [
    r"(?:ابن|طفل|ولد|بنت|رضيع).{0,35}(?:بلع|ابتلع|ابلع)",
    r"(?:بلع|ابتلع|ابلع).{0,25}(?:عملة|معدنية|حاجة|جسم|بطارية|مسمار|دبوس|coin)",
]

_RADIOLOGY_KEYWORDS_FLAT: list[str] = [
    "رنين", "مرنانة", "مرنانه", "mri", "رنين مغناطيسي",
    "سكانر", "مقطعية", "مقطعي", "مقطعيه", "ct",
    "سونار", "تلفزيوني", "تلفزيونية", "تلفزيونيه", "ultrasound",
    "دوبلر", "doppler",
    "ماموجرام", "mammogram",
    "ديكسا", "dexa", "هشاشة عظام", "كثافة عظام",
    "pet scan", "مسح ذري", "بت سكان",
    "بانوراما", "باناراما", "سيفالوميتريك",
    "إكس راي", "xray", "x-ray",
    "أشعة عادية", "اشعة", "اشعه", "أشعه", "أشعة",
    # [STABILITY] added missing variants that were detected in _MODALITY_KEYWORDS
    # but absent from the flat list, causing is_radiology_question() to return False
    # for questions containing these terms — leading to wrong RAG routing.
    "إيكو", "ايكو", "echo",
    "هولتر", "holter",
    "pet-scan", "بيتي سكان",
    "باناراما", "cephalometric",
]


def is_radiology_question(question: str) -> bool:
    """Return True when the question is about a radiology scan/procedure."""
    q = (question or "").lower()
    return any(kw in q for kw in _RADIOLOGY_KEYWORDS_FLAT)


def is_emergency_radiology(question: str) -> bool:
    """Return True when the question involves an emergency that needs immediate radiology (e.g. swallowed item)."""
    q = (question or "").lower()
    return any(re.search(p, q) for p in _EMERGENCY_PATTERNS)


def detect_modalities(question: str) -> list[str]:
    """Return list of modality keys detected in the question (ordered, most specific first)."""
    q = (question or "").lower()
    found: list[str] = []
    seen: set[str] = set()
    for keywords, modality in _MODALITY_KEYWORDS:
        if any(kw in q for kw in keywords) and modality not in seen:
            found.append(modality)
            seen.add(modality)
    return found


def _format_modality_context(m: dict) -> str:
    parts = [f"معلومات عن {m['arabic_name']}:"]
    if "prep" in m:
        parts.append(f"- التحضير: {m['prep']}")
    if "machine_type" in m:
        parts.append(f"- الجهاز: {m['machine_type']}")
    if "contraindications" in m:
        parts.append(f"- ملاحظات مهمة: {m['contraindications']}")
    if "duration" in m:
        parts.append(f"- المدة: {m['duration']}")
    if "gender_note" in m:
        parts.append(f"- ملاحظة: {m['gender_note']}")
    if "routing_note" in m:
        parts.append(f"- توجيه: {m['routing_note']}")
    if m.get("booking_required"):
        parts.append("- مهم: يحتاج حجز مسبق قبل 24-48 ساعة.")
    if m.get("same_day_result"):
        parts.append("- النتيجة: تطلع في نفس اليوم.")
    return "\n".join(parts)


def get_radiology_context(question: str) -> str:
    """
    Build a rich Arabic context string for radiology questions.
    Used when MCP has no pricing/schedule data for the requested scan.
    The returned text is injected as a Document for the QA engine.
    """
    modalities = detect_modalities(question)
    parts: list[str] = []

    for key in modalities:
        m = RADIOLOGY_MODALITIES.get(key)
        if m:
            parts.append(_format_modality_context(m))

    # Always append the call-to-hospital note for pricing
    parts.append(
        "الأسعار والحجز: السعر الدقيق لهذه الخدمة غير متوفر الآن في النظام. "
        "يرجى الاتصال باستقبال المستشفى مباشرةً للاستفسار عن السعر وتحديد موعد."
    )

    if not parts or len(parts) == 1:
        # No specific modality detected — give generic radiology guidance
        return (
            "معلومات عامة عن قسم الأشعة:\n"
            "- الأشعة العادية (X-Ray): لا تحتاج صيام، النتيجة في نفس اليوم.\n"
            "- الأشعة المقطعية (سكانر): صيام 4 ساعات لو بالصبغة.\n"
            "- الرنين المغناطيسي (MRI): لا صيام إلا لو بالصبغة، ممنوع لمرضى Pacemaker.\n"
            "- السونار (أشعة تلفزيونية): آمن تماماً، صيام للكبد والمرارة فقط.\n\n"
            "للأسعار والحجز: يرجى الاتصال باستقبال المستشفى مباشرةً."
        )

    return "\n\n".join(parts)


_EMERGENCY_RADIOLOGY_CONTEXT = (
    "[حالة طارئة — أشعة فورية]\n"
    "هذه حالة تستوجب الذهاب فوراً لقسم الطوارئ.\n"
    "قسم الطوارئ مجهز بجهاز أشعة عادية (X-Ray) داخله ويقدر يعمل الأشعة المطلوبة فوراً.\n"
    "لا تنتظر في قسم الأشعة العادي — روح الطوارئ دلوقتي."
)


def get_emergency_radiology_context() -> str:
    return _EMERGENCY_RADIOLOGY_CONTEXT