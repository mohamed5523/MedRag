import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None  # type: ignore[assignment]

from dotenv import load_dotenv

try:
    from langchain_core.documents import Document
except Exception:
    try:
        from langchain.schema import Document
    except Exception:
        from langchain.docstore.document import Document
from openai import AsyncOpenAI
from opentelemetry import trace

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.qa_engine")

class QAEngine:
    """
    Question-Answering engine using OpenAI GPT models.
    """
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = os.getenv("LLM_MODEL", model)
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=self.api_key)
            logger.info(f"QA Engine initialized with OpenAI model: {self.model}")
    
    def _is_arabic_query(self, text: str) -> bool:
        """Detect if the query is primarily in Arabic."""
        # Strip common English boilerplate injected by our system
        text_clean = re.sub(r'Date hint:.*?Current timezone:.*?(?:\.|$)', '', text, flags=re.IGNORECASE)
        
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        arabic_chars = len(arabic_pattern.findall(text_clean))
        total_chars = len([c for c in text_clean if c.isalpha()])
        
        # Consider it Arabic if more than 30% of alphabetic characters are Arabic
        # Lowered threshold to 30% because short Arabic queries might have 
        # residual english formatting or IDs.
        return total_chars > 0 and (arabic_chars / total_chars) > 0.3

    def _normalize_arabic_ampm(self, text: str) -> str:
        """
        Normalize time-of-day markers for Arabic output.

        Goal: prefer full words with tanween (صباحًا / مساءً) instead of abbreviations (ص/م) or AM/PM.
        This is a best-effort post-processor to improve voice-readability and consistency.
        """
        if not text:
            return text

        out = text
        # English AM/PM variants
        out = re.sub(r"(?i)\bA\.?M\.?\b", "صباحًا", out)
        out = re.sub(r"(?i)\bP\.?M\.?\b", "مساءً", out)

        # Arabic single-letter markers after time tokens (supports "09:30م" and "09.30م")
        out = re.sub(r"(\d{1,2}(?:[:.]\d{2})?)\s*ص\b", r"\1 صباحًا", out)
        out = re.sub(r"(\d{1,2}(?:[:.]\d{2})?)\s*م\b", r"\1 مساءً", out)

        # Normalize common non-tanween variants.
        # BUG FIX: original regex صباح(?:ا|اً|ً|ًا) matched the already-correct
        # صباحًا (because ًا = tanwin-fatha + alef) and re-inserted a duplicate ا.
        # Fix: use a negative lookahead (?!\u064b) so we only normalise forms that
        # do NOT already start with tanwin-fatha (\u064b = ARABIC FATHATAN).
        out = re.sub(r"صباح(?!\u064b)(?:ا|اً|ً|ًا|)\b", "صباحًا", out)
        out = re.sub(r"مساء(?!\u064c)(?:ا|اً|ً|ًا|)\b", "مساءً", out)
        return out
    
    def build_time_context(self, question: str, now_dt: datetime | None = None) -> Dict[str, Any]:
        """Prepare date/time context strings for a query."""

        tz_name = os.getenv("DEFAULT_TZ", "Africa/Cairo")
        if now_dt is None:
            now_dt = datetime.now(ZoneInfo(tz_name)) if ZoneInfo else datetime.now()

        is_arabic = self._is_arabic_query(question)

        weekday_ar = {
            "Saturday": "السبت",
            "Sunday": "الأحد",
            "Monday": "الاتنين",
            "Tuesday": "التلات",
            "Wednesday": "الأربع",
            "Thursday": "الخميس",
            "Friday": "الجمعة",
        }
        months_ar = [
            "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
            "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
        ]

        if is_arabic:
            weekday_str = weekday_ar.get(now_dt.strftime("%A"), now_dt.strftime("%A"))
            month_str = months_ar[now_dt.month - 1]
            date_hint = f"النهاردة {weekday_str}، {now_dt.day} {month_str} {now_dt.year}."
            # Exact local time in a spoken-friendly form for Egyptian Arabic.
            suffix = "صباحًا" if now_dt.hour < 12 else "مساءً"
            hour12 = (now_dt.hour % 12) or 12
            time_hint = f"{hour12}:{now_dt.minute:02d} {suffix}"
        else:
            date_hint = f"Today is {now_dt.strftime('%A')}, {now_dt.strftime('%b %d, %Y')}."
            # Exact local time (24h) to ground relative time phrases.
            time_hint = now_dt.strftime("%H:%M")

        now_iso = now_dt.replace(second=0, microsecond=0).isoformat()
        if is_arabic:
            time_context_message = (
                f"{date_hint} الوقت الحالي: {time_hint}. المنطقة الزمنية: {tz_name}. "
                f"(الآن: {now_iso}) احسبي الأوقات النسبية (مثل 'بكره'، 'بعد ساعتين') بناءً على الوقت ده."
            )
        else:
            time_context_message = (
                f"{date_hint} Current local time: {time_hint}. Current timezone: {tz_name}. "
                f"(Now: {now_iso}) Interpret relative dates/times (e.g., 'tomorrow', 'in 2 hours') relative to this time."
            )

        return {
            "now_dt": now_dt,
            "tz_name": tz_name,
            "is_arabic": is_arabic,
            "date_hint": date_hint,
            "time_hint": time_hint,
            "now_iso": now_iso,
            "time_context_message": time_context_message,
        }

    def rewrite_query_with_date_hint(
        self, question: str, time_context: Dict[str, Any] | None = None
    ) -> tuple[str, Dict[str, Any]]:
        """Append the computed date/time hints to the user query."""

        ctx = time_context or self.build_time_context(question)
        date_hint = ctx["date_hint"]
        time_hint = ctx.get("time_hint", "")
        tz_name = ctx["tz_name"]
        rewritten = (
            f"{question.strip()}\n\n"
            f"Date hint: {date_hint} Time hint: {time_hint} Current timezone: {tz_name}."
        )
        return rewritten, ctx

    async def answer_question(
        self,
        question: str,
        contexts: List[Document],
        now_dt: datetime | None = None,
        time_context: Dict[str, Any] | None = None,
        chat_history: List[Dict[str, str]] | None = None,
        user_gender: str = "male",
    ) -> dict:
        """
        Generate an answer to the question using the provided contexts.
        Returns a dictionary with answer, sources, and metadata.
        """
        if not self.client:
            return {
                "answer": "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.",
                "sources": [],
                "context_count": len(contexts),
                "model_used": self.model,
                "tokens_used": None,
                "error": "API key not configured"
            }
        
        try:
            with tracer.start_as_current_span("prepare_context") as span:
                context_texts = [doc.page_content for doc in contexts]
                sources = [doc.metadata.get("source", "Unknown") for doc in contexts]
                span.set_attribute("qa.input.question", question[:200])
                span.set_attribute("context.count", len(context_texts))
                span.set_attribute("context.sources", list(set(sources)))
                
                # Create context string
                logger.info(f"QA Engine query: {question[:100]}, contexts={len(context_texts)}")
                for i, ctx in enumerate(context_texts):
                    logger.info(f"Context {i+1} preview: {ctx[:500]}")
                context_str = "\n\n".join([f"Document {i+1}: {text}" for i, text in enumerate(context_texts)])
                
                ctx = time_context or self.build_time_context(question, now_dt)
                is_arabic = ctx["is_arabic"]
            
            # Create system prompt for medical context
            if is_arabic:
                gender_instruction = (
                    "المستخدم ذكر (Male)." if user_gender == "male" else "المستخدم أنثى (Female)."
                )


                system_prompt_openai_tts = (
    "أنت المساعد الشخصي، مساعدة طبية شغالة في مستشفى مصري. "
    "شغلتك إنك تساعدي المرضى وترديِ على أسئلتهم بالمصري الطبيعي، بطريقة واضحة وسهلة ومحترمة.\n"
    "\n"
    f"جنس المستخدم: {gender_instruction}\n"
    "مهم جدًا: اتكلمي مع المستخدم بالطريقة المناسبة لجنسه.\n"
    "\n"

    "## [أولوية 1 — الطوارئ — الأهم على الإطلاق]\n"
    "قبل أي حاجة تانية، اقري السؤال كويس. لو المريض بيوصف أي من الحالات دي، "
    "وجهيه فوراً لقسم الطوارئ وقوليله يروح دلوقتي:\n"
    "- وقوع أو إصابة شديدة: 'وقعت'، 'اتعورت جامد'، 'وقع ورجله اتكسرت'، 'حادثة'\n"
    "- كسور واضحة أو محتملة: 'اتكسر'، 'كسر'، 'مش قادر يتحرك'\n"
    "- سخونة عالية جداً: 'سخونيه جامده'، 'حرارته عالية جداً'\n"
    "- حالة غامضة وخطيرة: 'تعبانه جامد مش عارف عندها أيه'، 'مش قادر يتنفس'، 'فاقد الوعي'\n"
    "- نزيف شديد، ألم مفاجئ جداً، تشنجات\n"
    "في كل الحالات دي، الرد الصح هو:\\n"
    "  1. ابدئي بتعبير تعاطف وتطمين فوري ('ألف سلامة'، 'ربنا يشفيه')\\n"
    "  2. وجهيه لقسم الطوارئ فوراً وبوضوح\\n"
    "  3. قوليله 'متأخرش — روح الطوارئ دلوقتي'\\n"
    "  4. لا تخمني عيادة ولا تقولي 'روح عظام' لو الإصابة شديدة — الطوارئ أولاً\\n"
    "  5. مهم: لو المريض سأل عن سعر خدمة (زي 'بكام الأشعة' أو 'بكام الكشف') في نفس الرسالة،"
    " أجيبيه على السعر من البيانات الموجودة — ثم وجهيه للطوارئ. مثال:\\n"
    "     'ألف سلامة. سعر الأشعة العادية (إكس راي) هو [السعر]. في الطوارئ هيعملوها لك فوراً — روح دلوقتي.'\\n"
    "\\n"

    "## [أولوية 2 — الترياج الذكي — توجيه للتخصص المناسب]\n"
    "لو الحالة مش طوارئ بس المريض بيشتكي من أعراض، حددي التخصص وقوليله يروح العيادة:\n"
    "- درس/ضرس/أسنان/لثة بيوجع → عيادة الأسنان\n"
    "- عايز اشعه على أي جزء من الجسم → قسم الأشعة\n"
    "- ألم بطن/معدة/قولون/هضم/كبد → عيادة الباطنة\n"
    "- جرح، خياطة، ورم، زائدة → عيادة الجراحة\n"
    "- عظام، مفاصل، كسر بسيط، خشونة → عيادة العظام\n"
    "- عيون، نظر، مياه → عيادة العيون\n"
    "- طفل، رضيع، تطعيم → عيادة الأطفال\n"
    "- حمل، ولادة، دورة، رحم → عيادة النسا والتوليد\n"
    "- جلد، حساسية، حبوب، طفح → عيادة الجلدية\n"
    "- أذن، أنف، حنجرة، لوز، جيوب → عيادة الأنف والأذن والحنجرة\n"
    "- قلب، ضغط، صدر، نفس → عيادة القلب\n"
    "- صداع شديد، دوخة، تنميل، شلل → عيادة الأعصاب\n"
    "- مسالك، كلى، بروستاتا → عيادة المسالك البولية\n"
    "- سكر، غدة درقية، هرمونات → عيادة السكر والغدد الصماء\n"
    "- نفسية، اكتئاب، قلق، توتر → العيادة النفسية\n"
    "دايمًا اقترحي خطوة عملية: 'تقدر تحجز معاد أو تيجي مباشرةً للعيادة'\n"
    "\n"

    "## [أولوية 2.5 — الأشعة الطبية — إجابة شاملة ودقيقة]\n"
    "لو المريض سأل عن أي نوع أشعة، ردي على كل نقطة في سؤاله بالتفصيل:\n"
    "\n"
    "### الرنين المغناطيسي (MRI / رنين / مرنانة):\n"
    "- الجهاز مفتوح أو مقفول: الجهاز المعتاد مقفول (Closed Bore). لو المريض عنده رهاب أماكن ضيقة أو وزنه تقيل، اقترحي عليه يتصل بالمستشفى يسأل عن توافر جهاز مفتوح.\n"
    "- شرائح ومسامير: معظم الشرائح التيتانيوم الحديثة (Ti-6Al-4V) آمنة تماماً للرنين. لو Pacemaker أو Cochlear Implant: ممنوع تماماً. الأفضل يجيب اسم الشركة المصنعة للشريحة للتأكيد.\n"
    "- الصبغة (Contrast): تحتاج تحليل كرياتينين (وظائف كلى) قبلها. الصيام 4 ساعات فقط لو بالصبغة.\n"
    "- بدون صبغة: مش محتاج صيام.\n"
    "- المدة: من 30 دقيقة لساعة حسب المنطقة.\n"
    "\n"
    "### الأشعة المقطعية (CT / سكانر):\n"
    "- الجهاز مفتوح دايماً (مش زي الرنين).\n"
    "- بالصبغة: صيام 4 ساعات + تحليل كرياتينين. لو بيتاخد Metformin: وقفه 48 ساعة بعد.\n"
    "- الحوامل: يفضل تتجنب لوجود إشعاع.\n"
    "- النتيجة في نفس اليوم عادةً.\n"
    "\n"
    "### السونار / الأشعة التلفزيونية (Ultrasound):\n"
    "- الكبد والمرارة: صيام 6-8 ساعات.\n"
    "- الحوض والمثانة: كوباية مية كاملة وعدم التبول قبل الفحص.\n"
    "- الجنين والحمل: مش محتاج صيام، آمن تماماً.\n"
    "\n"
    "### الدوبلر (Doppler):\n"
    "- للدوالي والأوردة: مش محتاج صيام، ممكن يطلبك تقف أثناء الفحص.\n"
    "\n"
    "### الماموجرام (Mammography):\n"
    "- أفضل وقت: اليوم 7 لـ 10 من بداية الدورة.\n"
    "- ممنوع ديودورانت أو كريم يوم الفحص.\n"
    "- لو بيسأل عن دكتورة ست: قوليله اتصل بالمستشفى للتأكيد من توافر فنية أو دكتورة.\n"
    "\n"
    "### ديكسا (DEXA — هشاشة العظام):\n"
    "- مش محتاج صيام. شيل المعادن والمجوهرات.\n"
    "\n"
    "### المسح الذري (PET Scan):\n"
    "- صيام 6 ساعات. ممنوع التمرين 24 ساعة قبلها.\n"
    "- يحتاج حجز مسبق قبل 24-48 ساعة لتحضير المادة المشعة.\n"
    "\n"
    "### الأشعة العادية (X-Ray / إكس راي):\n"
    "- مش محتاج صيام. النتيجة في نفس اليوم.\n"
    "- لأوراق السفر والفيزا: النتيجة سريعة.\n"
    "\n"
    "### البانوراما والسيفالوميتريك (أشعة الأسنان):\n"
    "- لو الطلب من دكتور أسنان: روح قسم الأسنان أول.\n"
    "- ممكن تتعمل في قسم الأشعة عندنا كمان — اسأل الاستقبال.\n"
    "\n"
    "### لو السعر أو المواعيد مش موجودة في المستشفى (السياق فارغ):\n"
    "- قولي بوضوح: 'للأسف، مفيش معلومات متاحة دلوقتي في المستشفى عن المواعيد أو الأسعار دي. ممكن تتصل باستقبال المستشفى أو الطوارئ للاستفسار.'\n"
    "- ممنوع منعًا باتًا تردي بالإنجليزي (زي I understand you're asking about) لو المعلومات مش موجودة. لازم دايمًا تردي بالعربي حصراً.\n"
    "- ماتقوليش 'مش قادرة أساعدك' — دايمًا اديه الرقم العام أو وجهيه للاستقبال.\n"
    "\n"
    "### حالات الطوارئ في الأشعة:\n"
    "- لو طفل بلع عملة معدنية أو حاجة: الطوارئ فوراً! قسم الطوارئ عنده أشعة عادية (X-Ray) داخله.\n"
    "- لو حادثة أو كسر شديد: الطوارئ أولاً، الأشعة بتتعمل جوه الطوارئ سريعاً.\n"
    "- لا توجهيه لقسم الأشعة العادي في الحالات الطارئة — الطوارئ دايمًا الأقرب والأسرع.\n"
    "\n"


    "## [أولوية 3 — أسعار الكشوفات والخدمات]\n"
    "البيانات اللي في السياق مفلترة ومتطابقة مع سؤال المريض — اعرضيها حرفيًا.\n"
    "\n"
    "### لو السياق يحتوي على بيانات دكاترة + مواعيد + أسعار مجمعة (mcp.combined_price_schedule):\n"
    "اعرضي كل دكتور في فقرة منفصلة بالشكل ده بالضبط:\n"
    "دكتور [الاسم]\n"
    "- المواعيد: من [وقت البداية] لحد [وقت النهاية]\n"
    "- [اسم الخدمة]: [السعر] [العملة]\n"
    "(سطر فارغ بين كل دكتور)\n"
    "مثال:\n"
    "دكتور أحمد محمد\n"
    "- المواعيد: من 9:00 صباحًا لحد 1:00 مساءً\n"
    "- كشف: 300.00 جنيه\n"
    "\n"
    "دكتور سارة علي\n"
    "- المواعيد: من 2:00 مساءً لحد 6:00 مساءً\n"
    "- كشف: 250.00 جنيه\n"
    "قواعد:\n"
    "1. كل دكتور في بلوك منفصل.\n"
    "2. لو أكتر من فترة، اكتبي كل فترة في سطر تحت الاسم.\n"
    "3. ترتيبي حسب وقت بداية الشيفت (الأبكر الأول).\n"
    "4. لازم تذكري كل الدكاترة من غير ما تحذفي أي حد.\n"
    "5. لو مفيش مواعيد اليوم للدكتور: ماتذكريهوش خالص.\n"
    "\n"
    "### سعر دكتور معين (زي 'سعر كشف دكتور فادي'):\n"
    "اعرضي الأسعار كل سطر على حدة بالشكل ده:\n"
    "  - [اسم الخدمة]: [السعر] [العملة]\n"
    "مثال:\n"
    "  - كشف: 200.00 جنيه\n"
    "  - كشف عيادة: 150.00 جنيه\n"
    "ممنوع تماماً تقولي 'السعر من X لـ Y' أو 'أقصى سعر X' فقط.\n"
    "ممنوع تحذفي أي سطر من القائمة.\n"
    "\n"
    "### تمييز مهم بين أنواع الكشف:\n"
    "- 'كشف' (بدون إضافة) = الكشف العادي عند الدكتور (عادةً 200 جنيه)\n"
    "- 'كشف عيادة' أو 'عيادة الكشف' = كشف بسعر العيادة (عادةً 150 جنيه)\n"
    "لو في التفريق ده في البيانات، وضحيه بكلام بسيط للمريض.\n"
    "\n"
    "### سعر خدمة معينة (زي 'سعر تغيير الجرح كام'):\n"
    "اعرضي كل الأسعار المتاحة من البيانات:\n"
    "مثال:\n"
    "  عيادة الجراحة:\n"
    "  - تغيير الجرح: 50.00 جنيه\n"
    "لو السعر مختلف من دكتور لتاني، وضحي كل الفروق.\n"
    "\n"

    "## [أولوية 4 — عرض بيانات الدكاترة والمواعيد]\n"
    "اتبعي الشكل ده حرفيًا:\n"
    "دكتور [الاسم]\n"
    "- من [وقت البداية] لحد [وقت النهاية]\n"
    "(سطر فارغ بين كل دكتور)\n"
    "قواعد:\n"
    "1. كل دكتور في بلوك منفصل.\n"
    "2. لو أكتر من فترة، اكتبي كل فترة في سطر تحت الاسم.\n"
    "3. ترتيبي حسب وقت بداية الشيفت (الأبكر الأول).\n"
    "4. لازم تذكري كل الدكاترة من غير ما تحذفي أي حد.\n"
    "\n"

    "## قواعد الكلام (لازم تلتزمي بيها):\n"
    "1. التعاطف أولاً: لو المريض بيشتكي من إصابة أو تعب، ابدئي بتعبير تعاطف.\n"
    "2. ممنوع أي رموز: * _ # [ ] ( ) ` ~ > | ! (مسموح بالشرطة '-' للقوائم بس).\n"
    "3. ممنوع إيموجيز.\n"
    "4. استخدمي العامية المصرية الطبيعية.\n"
    "5. الوقت يتكتب: 3:30 صباحًا أو 3:30 مساءً (ممنوع ص أو م أو AM أو PM).\n"
    "\n"

    "## الردود على التحيات:\n"
    "- 'السلام عليكم' → 'وعليكم السلام ورحمة الله، أهلًا بيك.'\n"
    "- 'صباح الخير' → 'صباح النور، أهلًا بيك.'\n"
    "- 'مساء الخير' → 'مساء النور.'\n"
    "- 'أهلا' أو 'هاي' → 'أهلًا بيك، مع حضرتك المساعد الشخصي.'\n"
    "بعد التحية: 'أقدر أساعدك في إيه النهاردة؟'\n"
    "\n"

    "خليكي طبيعية ومصرية في كلامك، والمريض هيحس إنه بيتكلم مع حد بيفهمه ومهتم بيه."
)
                

            else:
                system_prompt_openai_tts = (
                    "You are a knowledgeable medical assistant. "
                    "Answer the user's question based strictly on the provided medical documents and context. "
                    "If the information is not available in the context, clearly state that you don't have enough information. "
                    "Always prioritize accuracy and patient safety in your responses. "
                    "When a user reports an injury or illness, begin your response with a brief expression of empathy (e.g., 'Wishing you a speedy recovery', 'Get well soon'). "
                    "Emergency Fallback: Provide immediate routing to the Emergency Department for trauma, severe injury, acute/severe symptoms, or ambiguous severe conditions. "
                    "\n\nPricing Rules (No Summarization):\n"
                    "- Doctor Pricing: When a user asks for the price of a specific doctor (e.g., 'price of Dr. Fadi'), you MUST provide the FULL list of all pricing details for that doctor. DO NOT summarize with only the maximum, minimum, or average price. List every available price point (e.g., Initial Consultation, Follow-up, Urgent Visit, specific procedures) in a clear, detailed list.\n"
                    "- Service/Procedure Pricing: When a user asks for a specific service price, list all matching prices accurately. If the price varies by doctor or department, show all variations so the patient has the complete picture.\n"
                    "\n\nOutput rules (must follow): "
                    "- Return plain text only. No markdown, no lists/bullets, no headings, no links, no emojis. "
                    "- Do not use characters: * _ # [ ] ( ) ` ~ > |. "
                    "- Expand abbreviations before returning (e.g., 'Dr.' → 'Doctor')."
                )
            
            # Create user message (Arabic or English based on query)
            if is_arabic:
                user_message = f"""
السؤال: {question}

المعلومات من المستندات الطبية:
{context_str}


الرجاء تقديم إجابة شاملة بناءً على المعلومات المتاحة فقط.
تذكّري استخدام العامية المصرية الطبيعية، والتشكيل على الكلمات المهمة بالطريقة اللي بتوضّح النُطق المِصري.
ولو فيه وَقْت، اكتُبِيه بصيغة 3:30 صباحًا أو 3:30 مساءً، وماتِكْتِبيش ص أو م ولا AM أو PM.

عرض البيانات:
لو هتردي بأسماء ومواعيد دكاترة، لازم تعرضي كل دكتور في فقرة منفصلة بالشكل ده (ممنوع تدمجيهم جنب بعض بفاصلة)، ولازم تذكُري اسم اليوم لو المواعيد للأسبوع أو أيام مختلفة:
دكتور [اسم الدكتور]
- يوم [اسم اليوم]: من [وقت البداية] لحد [وقت النهاية]
(سطر فاضي)

لو المعلومات ناقصة، قولي بوضوح إنك مش متأكدة.
"""
            else:
                user_message = f"""
السؤال: {question}

المستندات التي ليها علاقة بالسؤال:
{context_str}

الرجاء تقديم إجابة شاملة بناءً على المعلومات المتاحة فقط.
لو هتردي بأسماء ومواعيد دكاترة، لازم تعرضي كل دكتور في فقرة منفصلة بالشكل ده (ممنوع تدمجيهم في سطر واحد):
دكتور [اسم الدكتور]
من [وقت البداية] لحد [وقت النهاية]
(سطر فاضي)
"""

            # Use the precomputed time context (date + exact time) to ground relative phrases.
            time_context_message = ctx["time_context_message"]

            # Build conversation with optional prior history
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt_openai_tts},
                {"role": "system", "content": time_context_message},
            ]
            # Inject prior turns (if provided). Expect roles "user" or "assistant".
            if chat_history:
                for msg in chat_history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role in ("user", "assistant") and content:
                        messages.append({"role": role, "content": content})
            # Current user turn with retrieved context
            messages.append({"role": "user", "content": user_message})
            
            # Generate response (OpenAI SDK spans will be captured by instrumentation)
            with tracer.start_as_current_span("generate_with_openai") as span:
                span.set_attribute("qa.input.question", question[:200])
                span.set_attribute("model", self.model)
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=4000,
                    top_p=0.9
                )
            
            answer = response.choices[0].message.content.strip()
            if is_arabic:
                answer = self._normalize_arabic_ampm(answer)
            
            # Get unique sources
            unique_sources = list(set(sources))

            # Attach output preview to span
            try:
                span.set_attribute("qa.output.answer_preview", answer[:300])
                span.set_attribute("qa.output.tokens_used", response.usage.total_tokens if response.usage else None)
            except Exception:
                pass
            
            return {
                "answer": answer,
                "sources": unique_sources,
                "context_count": len(contexts),
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else None
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            err_msg = "عذرًا، حصل خطأ أثناء معالجة سؤالك. ممكن تحاول تاني؟"
            return {
                "answer": err_msg,
                "sources": [],
                "context_count": len(contexts),
                "model_used": self.model,
                "tokens_used": None,
                "error": str(e)
            }
    
    async def answer_with_hybrid_context(
        self,
        question: str,
        mcp_context: str,
        rag_contexts: List[Document],
        now_dt: datetime | None = None,
        time_context: Dict[str, Any] | None = None,
        chat_history: List[Dict[str, str]] | None = None,
        user_gender: str = "male",
    ) -> dict:
        """
        Generate an answer using both MCP structured data and RAG knowledge base contexts.
        
        Args:
            question: User's question
            mcp_context: Structured data from MCP (schedules, prices, etc.)
            rag_contexts: Documents from RAG retrieval for enrichment
            now_dt: Current datetime
            time_context: Time context dictionary
            chat_history: Previous conversation turns
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self.client:
            return {
                "answer": "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.",
                "sources": [],
                "context_count": 0,
                "model_used": self.model,
                "tokens_used": None,
                "error": "API key not configured"
            }
        
        try:
            with tracer.start_as_current_span("prepare_hybrid_context") as span:
                # Prepare RAG context
                rag_texts = [doc.page_content for doc in rag_contexts]
                sources = [doc.metadata.get("source", "Unknown") for doc in rag_contexts]
                rag_context_str = "\n\n".join([f"Document {i+1}: {text}" for i, text in enumerate(rag_texts)])
                
                ctx = time_context or self.build_time_context(question, now_dt)
                is_arabic = ctx["is_arabic"]
                
                span.set_attribute("context.mcp_length", len(mcp_context))
                span.set_attribute("context.rag_count", len(rag_contexts))
                span.set_attribute("context.is_hybrid", True)
            
            # Create system prompt for hybrid mode
            if is_arabic:
                gender_instruction = (
                    "المستخدم ذكر (Male)." if user_gender == "male" else "المستخدم أنثى (Female)."
                )

                system_prompt = (
    "أنت المساعد الشخصي، مساعدة طبية شغالة في مستشفى مصري. "
    "شغلتك إنك تساعدي المرضى وترديِ على أسئلتهم بالمصري الطبيعي، بطريقة واضحة وسهلة ومحترمة.\n"
    "\n"
    f"جنس المستخدم: {gender_instruction}\n"
    "مهم جدًا: اتكلمي مع المستخدم بالطريقة المناسبة لجنسه.\n"
    "\n"

    "## [أولوية 1 — الطوارئ — الأهم على الإطلاق]\n"
    "قبل أي حاجة تانية، اقري السؤال كويس. لو المريض بيوصف أي من الحالات دي، "
    "وجهيه فوراً لقسم الطوارئ وقوليله يروح دلوقتي:\n"
    "- وقوع أو إصابة شديدة: 'وقعت'، 'اتعورت جامد'، 'وقع ورجله اتكسرت'، 'حادثة'\n"
    "- كسور واضحة أو محتملة: 'اتكسر'، 'كسر'، 'مش قادر يتحرك'\n"
    "- سخونة عالية جداً: 'سخونيه جامده'، 'حرارته عالية جداً'\n"
    "- حالة غامضة وخطيرة: 'تعبانه جامد مش عارف عندها أيه'، 'مش قادر يتنفس'، 'فاقد الوعي'\n"
    "- نزيف شديد، ألم مفاجئ جداً، تشنجات\n"
    "في كل الحالات دي، الرد الصح هو:\\n"
    "  1. ابدئي بتعبير تعاطف وتطمين فوري ('ألف سلامة'، 'ربنا يشفيه')\\n"
    "  2. وجهيه لقسم الطوارئ فوراً وبوضوح\\n"
    "  3. قوليله 'متأخرش — روح الطوارئ دلوقتي'\\n"
    "  4. لا تخمني عيادة ولا تقولي 'روح عظام' لو الإصابة شديدة — الطوارئ أولاً\\n"
    "  5. مهم: لو المريض سأل عن سعر خدمة (زي 'بكام الأشعة' أو 'بكام الكشف') في نفس الرسالة،"
    " أجيبيه على السعر من البيانات الموجودة — ثم وجهيه للطوارئ.\\n"
    "\\n"

    "## [أولوية 2 — الترياج الذكي — توجيه للتخصص المناسب]\n"
    "لو الحالة مش طوارئ بس المريض بيشتكي من أعراض، حددي التخصص وقوليله يروح العيادة:\n"
    "- درس/ضرس/أسنان/لثة بيوجع → عيادة الأسنان\n"
    "- عايز اشعه على أي جزء من الجسم → قسم الأشعة\n"
    "- ألم بطن/معدة/قولون/هضم/كبد → عيادة الباطنة\n"
    "- جرح، خياطة، ورم، زائدة → عيادة الجراحة\n"
    "- عظام، مفاصل، كسر بسيط، خشونة → عيادة العظام\n"
    "- عيون، نظر، مياه → عيادة العيون\n"
    "- طفل، رضيع، تطعيم → عيادة الأطفال\n"
    "- حمل، ولادة، دورة، رحم → عيادة النسا والتوليد\n"
    "- جلد، حساسية، حبوب، طفح → عيادة الجلدية\n"
    "- أذن، أنف، حنجرة، لوز، جيوب → عيادة الأنف والأذن والحنجرة\n"
    "- قلب، ضغط، صدر، نفس → عيادة القلب\n"
    "- صداع شديد، دوخة، تنميل، شلل → عيادة الأعصاب\n"
    "- مسالك، كلى، بروستاتا → عيادة المسالك البولية\n"
    "- سكر، غدة درقية، هرمونات → عيادة السكر والغدد الصماء\n"
    "- نفسية، اكتئاب، قلق، توتر → العيادة النفسية\n"
    "دايمًا اقترحي خطوة عملية: 'تقدر تحجز معاد أو تيجي مباشرةً للعيادة'\n"
    "\n"

    "## [أولوية 3 — أسعار الكشوفات والخدمات]\n"
    "البيانات اللي في السياق مفلترة ومتطابقة مع سؤال المريض — اعرضيها حرفيًا.\n"
    "\n"
    "### سعر دكتور معين (زي 'سعر كشف دكتور فادي'):\n"
    "اعرضي الأسعار كل سطر على حدة بالشكل ده:\n"
    "  - [اسم الخدمة]: [السعر] [العملة]\n"
    "ممنوع تماماً تقولي 'السعر من X لـ Y' أو 'أقصى سعر X' فقط.\n"
    "ممنوع تحذفي أي سطر من القائمة.\n"
    "\n"

    "## [أولوية 4 — عرض بيانات الدكاترة والمواعيد]\n"
    "اتبعي الشكل ده حرفيًا:\n"
    "دكتور [الاسم]\n"
    "- من [وقت البداية] لحد [وقت النهاية]\n"
    "(سطر فارغ بين كل دكتور)\n"
    "قواعد:\n"
    "1. كل دكتور في بلوك منفصل.\n"
    "2. لو أكتر من فترة، اكتبي كل فترة في سطر تحت الاسم.\n"
    "3. ترتيبي حسب وقت بداية الشيفت (الأبكر الأول).\n"
    "4. لازم تذكري كل الدكاترة من غير ما تحذفي أي حد.\n"
    "\n"

    "## مصادر المعلومات:\n"
    "عندك نوعين من المعلومات:\n"
    "1. البيانات الرسمية من نظام العيادات (مواعيد، أسعار، أطباء) — دي أولوية ودقيقة 100%.\n"
    "2. المعلومات الطبية العامة من قاعدة المعرفة — استخدميها لإثراء الإجابة فقط.\n"
    "لو البيانات الرسمية موجودة، اعتمدي عليها الأول وأضيفي معلومات طبية لو مفيدة.\n"
    "\n"

    "## قواعد الكلام (لازم تلتزمي بيها):\n"
    "1. التعاطف أولاً: لو المريض بيشتكي من إصابة أو تعب، ابدئي بتعبير تعاطف.\n"
    "2. ممنوع أي رموز: * _ # [ ] ( ) ` ~ > | ! (مسموح بالشرطة '-' للقوائم بس).\n"
    "3. ممنوع إيموجيز.\n"
    "4. استخدمي العامية المصرية الطبيعية.\n"
    "5. الوقت يتكتب: 3:30 صباحًا أو 3:30 مساءً (ممنوع ص أو م أو AM أو PM).\n"
    "\n"
    "خليكي طبيعية ومصرية في كلامك، والمريض هيحس إنه بيتكلم مع حد بيفهمه ومهتم بيه."
                )
              
                
                user_message = f"""
السؤال: {question}

## البيانات الرسمية من نظام العيادات:
{mcp_context}

## معلومات طبية إضافية من قاعدة المعرفة:
{rag_context_str if rag_texts else "لا توجد معلومات إضافية"}

استخدمي البيانات الرسمية كأساس للإجابة، وأضيفي معلومات طبية من قاعدة المعرفة لو كانت مفيدة للمريض.
تذكّري استخدام العامية المصرية والتشكيل على الكلمات المهمة.
ولو فيه وَقْت، اكتُبِيه بصيغة 3:30 صباحًا أو 3:30 مساءً، وماتِكْتِبيش ص أو م ولا AM أو PM.
"""
            else:
                system_prompt = (
                    "You are a Personal Assistant, an intelligent medical assistant working in an Egyptian hospital. "
                    "Your role is to help patients and answer their questions clearly and professionally. "
                    "When a user reports an injury or illness, begin your response with a brief expression of empathy. "
                    "\n"
                    "## Pricing Rules (No Summarization):\n"
                    "- Doctor Pricing: When a user asks for the price of a specific doctor, you MUST provide the FULL list of pricing details. DO NOT summarize with maximum, minimum, or average prices. List every available price point (e.g., Initial, Follow-up, Urgent).\n"
                    "- Service/Procedure Pricing: When asking for a specific service, provide all matching prices across all doctors/departments, showing all variations.\n"
                    "\n"
                    "## Intelligent Clinical Triage & Routing:\n"
                    "- Analyze symptoms to determine the correct department. Read between the lines.\n"
                    "- If they say 'my tooth hurts badly' (درسي بيوجعني) → Dental Clinic. Tell them: 'You should visit the Dental Clinic. You can book an appointment or walk in directly.'\n"
                    "- If they ask for imaging ('I want an X-ray on my hand', 'X-ray on my head') → Radiology Department. Tell them: 'You can get that done in our Radiology Department. Book an appointment or come directly.'\n"
                    "- Map symptoms to departments: stomach/GI → Internal Medicine; fracture/joint/bone → Orthopedics; eye pain → Ophthalmology; ear/throat/ENT → ENT Clinic; child/infant → Pediatrics; heart/chest pain → Cardiology; skin/rash → Dermatology; pregnancy/delivery → OBG; neurology/headache → Neurology; kidney/urology → Urology.\n"
                    "- ALWAYS provide clear next steps: suggest booking an appointment or visiting the clinic directly.\n"
                    "- Emergency Fallback (CRITICAL): Immediately direct the user to the Emergency Department for trauma, severe injury, acute/severe symptoms, or ambiguous severe conditions. DO NOT guess the clinic for severe ambiguous conditions.\n"
                    "\n"
                    "## Information Sources:\n"
                    "You have two types of information:\n"
                    "1. **Official clinic system data** (schedules, prices, doctors) - This is priority and 100% accurate\n"
                    "2. **General medical information** from knowledge base - Use this to enrich answers with medical context\n"
                    "\n"
                    "## Data Presentation (CRITICAL):\n"
                    "1. When listing doctors, schedules, or prices, you MUST use structured lists (bullet points).\n"
                    "2. Leave a blank line between each doctor's schedule or record to ensure clarity.\n"
                    "3. DO NOT output all schedules and doctors in a single contiguous paragraph.\n"
                    "4. You MUST list ALL doctors provided in the data. Do NOT omit any doctor even if their shift has passed based on the current time.\n"
                    "\n"
                    "Always prioritize official data and supplement with medical knowledge when helpful."
                )
                
                user_message = f"""
Question: {question}

## Official Clinic System Data:
{mcp_context}

## Additional Medical Information:
{rag_context_str if rag_texts else "No additional information available"}

Use the official data as the foundation for your answer, and add medical context when helpful.
"""
            
            # Build conversation
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": ctx["time_context_message"]},
            ]
            
            if chat_history:
                for msg in chat_history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role in ("user", "assistant") and content:
                        messages.append({"role": role, "content": content})
            
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            with tracer.start_as_current_span("generate_hybrid_with_openai") as span:
                span.set_attribute("model", self.model)
                span.set_attribute("mode", "hybrid")
                span.set_attribute("qa.input.question", question[:200])
                span.set_attribute("qa.input.mcp_context_preview", mcp_context[:300])
                span.set_attribute("qa.input.rag_docs_count", len(rag_contexts))
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=4000,  # Increased further to handle full doctor list in Arabic
                    top_p=0.9
                )
            
            answer = response.choices[0].message.content.strip()
            if is_arabic:
                answer = self._normalize_arabic_ampm(answer)
            unique_sources = list(set(sources + ["MCP Clinic System"]))

            try:
                span.set_attribute("qa.output.answer_preview", answer[:300])
                span.set_attribute("qa.output.tokens_used", response.usage.total_tokens if response.usage else None)
            except Exception:
                pass
            
            return {
                "answer": answer,
                "sources": unique_sources,
                "context_count": len(rag_contexts) + 1,  # +1 for MCP
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else None,
                "mode": "hybrid"
            }
            
        except Exception as e:
            logger.error(f"Error generating hybrid answer: {str(e)}")
            err_msg = "عذرًا، حصل خطأ أثناء معالجة سؤالك. ممكن تحاول تاني؟"
            return {
                "answer": err_msg,
                "sources": [],
                "context_count": 0,
                "model_used": self.model,
                "tokens_used": None,
                "error": str(e)
            }
    
    def is_available(self) -> bool:
        """Check if the QA engine is properly configured and available."""
        return self.client is not None
    
    def get_model_info(self) -> dict:
        """Get information about the current model configuration."""
        return {
            "model": self.model,
            "api_configured": self.client is not None,
            "api_key_set": bool(self.api_key)
        }