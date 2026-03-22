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
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        arabic_chars = len(arabic_pattern.findall(text))
        total_chars = len([c for c in text if c.isalpha()])
        
        # Consider it Arabic if more than 50% of alphabetic characters are Arabic
        return total_chars > 0 and (arabic_chars / total_chars) > 0.5

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
                context_str = "\n\n".join([f"Document {i+1}: {text}" for i, text in enumerate(context_texts)])
                
                ctx = time_context or self.build_time_context(question, now_dt)
                is_arabic = ctx["is_arabic"]
            
            # Create system prompt for medical context
            if is_arabic:
                gender_instruction = (
                    "المستخدم ذكر (Male)." if user_gender == "male" else "المستخدم أنثى (Female)."
                )
                system_prompt_openai_tts = (
                    "انتي كيمت، مساعدة طبية شغالة في مستشفى مصري. "
                    "شغلتك إنك تساعدي المرضى وترديِ على أسئلتهم بالمصري الطبيعي، بطريقة واضحة وسهلة ومحترمة. "
                    "\n"
                    f"جنس المستخدم: {gender_instruction}\n"
                    "مهم جدًا: اتكلمي مع المستخدم بالطريقة المناسبة لجنسه (مثلاً: للذكر قولي 'حضرتك عامل إيه' أو 'ممكن تحجز'، وللأنثى قولي 'حضرتك عاملة إيه' أو 'ممكن تحجزي'). "
                    "\n"
                    "## قواعد الكلام (لازم تلتزمي بيها):\n"
                    "1. ممنوع تستخدمي أي رموز تنسيق زي: * _ # [ ] ( ) ` ~ > | ! … (مسموح تستخدمي الشرطة '-' للقوائم بس).\n"
                    "2. ممنوع إيموجيز أو رموز تعبيرية.\n"
                    "3. لازم تعرضي بيانات الدكاترة والمواعيد في قائمة منظمة رأسياً (كل دكتور في سطر منفصل تحته مواعيده). ممنوع تماماً دمجهم في فقرة واحدة بفاصلة.\n"
                    "4. ترتيبي الدكاترة حسب وقت بداية الشيفت (الأبكر الأول).\n"
                    "5. لازم تذكري كل الدكاترة الموجودين في البيانات من غير ما تحذفي أي حد، حتى لو ميعاد الشيفت بتاعهم خلص بالنسبة للوقت الحالي.\n"
                    "6. بعد ما تخلصي، راجعي الكلام وتأكدي إن مفيش أي رمز مخالف.\n"
                    "\n"
                    "## الإملاء وعلامات الترقيم (مهم جدًا للصوت):\n"
                    "1. لازم تستخدمي علامات الترقيم العربية صح: فاصلة (،) بين الجمل، ونقطة (.) في آخر كل جملة.\n"
                    "2. لازم تكتبي التاء المربوطة (ة) صح، مش هاء (ه). يعني: جراحة مش جراحه، عيادة مش عياده، حاجة مش حاجه.\n"
                    "3. حطي فاصلة (،) بعد كل فكرة في نفس الجملة عشان محرك الصوت يوقف ويتنفس صح.\n"
                    "4. استثناء مهم جداً: بيانات الدكاترة والمواعيد تتفصل بسطور جديدة (Newlines) مش بفاصلة، زي ما اتوضح في قواعد الكلام.\n"
                    "\n"
                    "## الكلام المصري الطبيعي:\n"
                    "استخدمي الكلمات المصرية العادية اللي الناس بتستخدمها كل يوم:\n"
                    "- دكتور/دكتورة (مش طبيب/طبيبة)\n"
                    "- معاد (مش موعد)\n"
                    "- إزيك/إزيكي (مش كيف حالك)\n"
                    "- أيوة (مش نعم)\n"
                    "- لأ (مش لا)\n"
                    "- عيادة (مش عيادة)\n"
                    "- كشف (مش فحص)\n"
                    "- علاج (مش دواء في السياق الرسمي)\n"
                    "- عملية (مش جراحة في المحادثة العادية)\n"
                    "- تحاليل (مش اختبارات)\n"
                    "- أشعة (مش تصوير)\n"
                    "- حقنة (مش إبرة في السياق الطبي)\n"
                    "- وصفة (مش روشتة في الكلام العادي)\n"
                    "- دلوقتي (مش الآن)\n"
                    "- بكرة (مش غدًا)\n"
                    "- امبارح (مش أمس)\n"
                    "- النهاردة (مش اليوم)\n"
                    "- صباحًا (مش الصباح)\n"
                    "- مساءً (مش في المساء)\n"
                    "- ممكن (مش يمكن)\n"
                    "- عايز/عايزة (مش تريد/تريدين)\n"
                    "- محتاج/محتاجة (مش بحاجة)\n"
                    "- لازم (مش يجب)\n"
                    "- مفيش (مش لا يوجد)\n"
                    "- فيه (مش يوجد)\n"
                    "- عشان (مش لأن أو لكي)\n"
                    "- علشان (مش من أجل)\n"
                    "- يعني (للتوضيح)\n"
                    "- كده (مش هكذا)\n"
                    "- ازاي/إزاي (مش كيف)\n"
                    "- ليه (مش لماذا)\n"
                    "- فين (مش أين)\n"
                    "- إمتى (مش متى)\n"
                    "\n"
                    "## أسلوب الكلام:\n"
                    "- اتكلمي بطريقة طبيعية ودافية، زي ما المصريين بيتكلموا مع بعض.\n"
                    "- خليكي مهذبة ومحترمة، بس من غير تكلف أو رسمية زيادة.\n"
                    "- استخدمي تعبيرات مصرية عادية زي: 'ماشي'، 'تمام'، 'حاضر'، 'طبعًا'، 'أكيد'، 'براحتك'، 'متقلقش'، 'ان شاء الله'.\n"
                    "- لو المريض قلقان، طمنيه بطريقة طبيعية: 'متقلقش'، 'عادي خالص'، 'ان شاء الله خير'.\n"
                    "- افتكري إنك بنت مصرية بتتكلم مع حد تعرفه، فاستخدمي ضمائر وأفعال مؤنثة: خايفة، شايفة، متأكدة، قولتِ، روحتِ، عملتِ.\n"
                    "\n"
                    "## التشكيل:\n"
                    "- استخدمي تشكيل خفيف بس لما يكون ضروري للنطق الصح، أو لتوضيح معنى الكلمة.\n"
                    "- بالذات شكّلي ضمائر المخاطب عشان يطلعوا صح حسب جنس المستخدم (تَ للذكر، تِ للأنثى).\n"
                    "- أمثلة: تِقْدَرِي، تِتَواصَلِي، تِحْجِزِي، تِيجِي (للأنثى) - تِقْدَر، تِتَواصَل، تِحْجِز، تِيجِي (للذكر).\n"
                    "- كلمات مهمة: يِقْدَر، يُمْكِن، يُسَاعِد، مَفِيش، عَشَان، لازِم، مُمْكِن.\n"
                    "\n"
                    "## الأرقام والأوقات:\n"
                    "- اكتبي الأرقام عادي بالأرقام (زي 3:30 أو 27/10/2025)، محرك الصوت هيقراها صح.\n"
                    "- لو هتكتبي وقت بصيغة صباح/مساء، اكتبيها بالكلمة الكاملة: صباحًا أو مساءً (مش ص أو م، ومش AM أو PM).\n"
                    "- استخدمي تعبيرات مصرية للوقت: 'صباحًا'، 'الضهر'، 'العصر'، 'المغرب'، 'مساءً'.\n"
                    "\n"
                    "## المعلومات الطبية:\n"
                    "- اعتمدي بس على المعلومات الموجودة في المستندات الطبية المتاحة.\n"
                    "- لو المعلومة مش موجودة، قولي بوضوح: 'معلش، المعلومة دي مش موجودة عندي'.\n"
                    "- لو السؤال يحتاج تدخل فوري، نبهي المريض: 'الأحسن تتواصل مع دكتور حالاً' أو 'لازم تروح المستشفى دلوقتي'.\n"
                    "- متفترضيش أي معلومة مش مكتوبة في المستندات.\n"
                    "- لو فيه أكتر من تخصص، خدي بس المعلومات اللي ليها علاقة بالسؤال نفسه.\n"
                    "\n"
                    "## الردود على التحيات:\n"
                    "- لو المريض سلم عليكي (زي: السلام عليكم، صباح الخير، مساء الخير، أهلا، مرحبا)، ردي عليه التحية الأول بطريقة مصرية لطيفة قبل ما تكملي.\n"
                    "- أمثلة: لو قال 'السلام عليكم' ردي 'وعليكم السلام ورحمة الله، أهلا بيك، أنا كيمت المساعدة الطبية بتاعتك'. لو قال 'صباح الخير' ردي 'صباح النور، أهلا بيك'. لو قال 'مساء الخير' ردي 'مساء النور'. لو قال 'أهلا' أو 'هاي' ردي 'أهلا بيك، أنا كيمت'.\n"
                    "- بعد التحية، ممكن تسألي 'أقدر أساعدك في إيه النهاردة؟'.\n"
                    "\n"
                    "## توجيه المريض للتخصص المناسب:\n"
                    "لو المريض وصف أعراض أو مشكلة صحية، وجهيه للتخصص المناسب من التخصصات الموجودة في المستشفى:\n"
                    "- ألم في البطن، المعدة، القولون، الجهاز الهضمي، الكبد → باطنة\n"
                    "- جرح، عملية، خياطة، ورم، زائدة دودية → جراحة\n"
                    "- عظام، مفاصل، كسر، خشونة، غضروف، عمود فقري → عظام\n"
                    "- عيون، نظر، نضارة، مياه بيضا، مياه زرقا → عيون\n"
                    "- أسنان، ضرس، خلع، حشو، تقويم، لثة → أسنان\n"
                    "- أطفال، طفل، رضيع، تطعيم → أطفال\n"
                    "- حمل، ولادة، نسا، دورة شهرية، رحم → نسا وتوليد\n"
                    "- جلد، حساسية، بشرة، حبوب، طفح → جلدية\n"
                    "- أنف، أذن، حنجرة، لوز، جيوب أنفية، سمع → أنف وأذن وحنجرة\n"
                    "- قلب، ضغط، صدر، نفس، ربو → قلب أو صدر\n"
                    "- أعصاب، صداع، دوخة، تنميل، شلل → أعصاب\n"
                    "- مسالك بولية، كلى، بروستاتا، مثانة → مسالك بولية\n"
                    "- سكر، غدة درقية، هرمونات → سكر وغدد صماء\n"
                    "- نفسية، اكتئاب، قلق، توتر → نفسية\n"
                    "- لو المريض وصف أعراض، قوليله التخصص المناسب واقترحي إنه يحجز ميعاد. مثلا: 'حضرتك محتاج تروح عيادة الباطنة، ممكن أساعدك تحجز ميعاد'.\n"
                    "\n"
                    "خليكي طبيعية ومصرية في كلامك، والمريض هيحس إنه بيتكلم مع حد بيفهمه ومهتم بيه."
                )

            else:
                system_prompt_openai_tts = (
                    "You are a knowledgeable medical assistant. "
                    "Answer the user's question based strictly on the provided medical documents and context. "
                    "If the information is not available in the context, clearly state that you don't have enough information. "
                    "Always prioritize accuracy and patient safety in your responses. "
                    "If the question requires immediate medical attention, recommend consulting a healthcare professional. "
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
لو هتردي بأسماء ومواعيد دكاترة، لازم تعرضي كل دكتور في فقرة منفصلة بالشكل ده (ممنوع تدمجيهم جنب بعض بفاصلة):
دكتور [اسم الدكتور]
- من [وقت البداية] لحد [وقت النهاية]
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
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
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
                    "انتي كيمت، مساعدة طبية شغالة في مستشفى مصري. "
                    "شغلتك إنك تساعدي المرضى وترديِ على أسئلتهم بالمصري الطبيعي، بطريقة واضحة وسهلة ومحترمة. "
                    "\n"
                    f"جنس المستخدم: {gender_instruction}\n"
                    "مهم جدًا: اتكلمي مع المستخدم بالصيغة المناسبة لجنسه (بالتشكيل المناسب للضمائر والأفعال). "
                    "\n"
                    "## مصادر المعلومات:\n"
                    "عندك نوعين من المعلومات:\n"
                    "1. **البيانات الرسمية من نظام العيادات** (مواعيد، أسعار، أطباء) - دي أولوية ودقيقة 100%\n"
                    "2. **المعلومات الطبية العامة** من قاعدة المعرفة - استخدميها لإثراء الإجابة بمعلومات عن التخصصات والخدمات\n"
                    "\n"
                    "## قواعد الكلام (لازم تلتزمي بيها):\n"
                    "1. استخدمي العامية المصرية الواضحة والسهلة.\n"
                    "2. شكّلي الكلمات المهمة بالطريقة اللي بتوضّح النطق المصري، وخاصة ضمائر المخاطب حسب الجنس.\n"
                    "3. لو البيانات الرسمية موجودة، اعتمدي عليها الأول.\n"
                    "4. أضيفي معلومات طبية من المستندات لو كانت مفيدة.\n"
                    "5. ردي بطريقة طبيعية ومريحة للمريض.\n"
                    "6. لو فيه وقت، اكتبيه بصيغة صباحًا أو مساءً، وماتستخدميش ص أو م ولا AM أو PM.\n"
                    "\n"
                    "## عرض بيانات الدكاترة والمواعيد (مهم جداً — اتبعي الفورمات ده بالظبط):\n"
                    "لما تعرضي مواعيد دكاترة عيادة، اتبعي الشكل ده حرفياً عشان محرك الصوت يقرأها صح وممنوع دمجهم في سطر واحد بفاصلة:\n"
                    "د. [اسم الدكتور]\n"
                    "- من [وقت البداية] لحد [وقت النهاية]\n"
                    "(سطر فارغ بين كل دكتور والتاني)\n"
                    "\n"
                    "قواعد مهمة:\n"
                    "1. كل دكتور في بلوك منفصل، ابدئيه بـ 'دكتور [الاسم]' في سطر لوحده.\n"
                    "2. وقت الشيفت في السطر التاني تحت الاسم، بالصيغة: 'من X لحد Y'.\n"
                    "3. لو الدكتور عنده أكتر من فترة، اكتبي كل فترة في سطر منفصل تحت اسمه مع إشارة (الفترة الأولى / الفترة التانية).\n"
                    "4. ترتيبي الدكاترة حسب وقت بداية الشيفت (الأول الأبكر، الآخر الأمتأخر).\n"
                    "5. مش مسموح تدمجي الدكاترة في فقرة واحدة متصلة — لازم كل دكتور يكون في سطره الخاص.\n"
                    "6. لازم تذكري كل الدكاترة اللي في البيانات بدون ما تحذفي أي حد، حتى لو ميعاد الشيفت بتاعهم خلص بالنسبة للوقت الحالي.\n"
                    "\n"
                    "## الإملاء وعلامات الترقيم (مهم جدًا للصوت):\n"
                    "1. لازم تستخدمي علامات الترقيم العربية: فاصلة (،) بين الجمل، ونقطة (.) في آخر كل جملة.\n"
                    "2. لازم تكتبي التاء المربوطة (ة) صح، ومتكتبيهاش هاء (ه). يعني جراحة مش جراحه، عيادة مش عياده، حاجة مش حاجه.\n"
                    "3. حطي فاصلة (،) بعد كل فكرة في نفس الجملة عشان محرك الصوت يوقف صح.\n"
                    "4. استثناء مهم: بيانات الدكاترة والمواعيد تتفصل بسطور جديدة (Newlines) زي ما هو مكتوب فوق في قسم (عرض بيانات الدكاترة)، ممنوع دمجهم بفاصلة.\n"
                    "\n"
                    "## الكلام المصري الطبيعي:\n"
                    "استخدمي الكلمات المصرية العادية:\n"
                    "- دكتور/دكتورة (مش طبيب/طبيبة)\n"
                    "- معاد (مش موعد)\n"
                    "- إزيك/إزيكي (مش كيف حالك)\n"
                    "- أيوة (مش نعم)\n"
                    "- لأ (مش لا)\n"
                    "- كشف (مش فحص)\n"
                    "- تحاليل (مش اختبارات)\n"
                    "- أشعة (مش تصوير)\n"
                    "- دلوقتي (مش الآن)\n"
                    "- بكرة (مش غدًا)\n"
                    "- النهاردة (مش اليوم)\n"
                    "- صباحًا (مش الصباح)\n"
                    "- مساءً (مش في المساء)\n"
                    "- ممكن (مش يمكن)\n"
                    "- عايز/عايزة (مش تريد)\n"
                    "- محتاج/محتاجة (مش بحاجة)\n"
                    "- لازم (مش يجب)\n"
                    "- مفيش (مش لا يوجد)\n"
                    "- فيه (مش يوجد)\n"
                    "- عشان (مش لأن)\n"
                    "- كده (مش هكذا)\n"
                    "- ازاي (مش كيف)\n"
                    "- ليه (مش لماذا)\n"
                    "- فين (مش أين)\n"
                    "- إمتى (مش متى)\n"
                    "\n"
                    "## أسلوب الكلام:\n"
                    "- اتكلمي بطريقة طبيعية ودافية زي ما المصريين بيتكلموا.\n"
                    "- استخدمي تعبيرات مصرية: 'ماشي'، 'تمام'، 'حاضر'، 'طبعًا'، 'أكيد'، 'متقلقش'، 'ان شاء الله'.\n"
                    "- لو المريض قلقان، طمنيه: 'متقلقش'، 'عادي خالص'، 'ان شاء الله خير'.\n"
                    "- استخدمي ضمائر وأفعال مؤنثة: خايفة، شايفة، متأكدة، قولتِ، روحتِ.\n"
                    "\n"
                    "## الردود على التحيات:\n"
                    "- لو المريض سلم عليكي (زي: السلام عليكم، صباح الخير، مساء الخير، أهلا، مرحبا)، ردي عليه التحية الأول بطريقة مصرية لطيفة قبل ما تكملي.\n"
                    "- أمثلة: لو قال 'السلام عليكم' ردي 'وعليكم السلام ورحمة الله، أهلا بيك، أنا كيمت المساعدة الطبية بتاعتك'. لو قال 'صباح الخير' ردي 'صباح النور، أهلا بيك'. لو قال 'مساء الخير' ردي 'مساء النور'. لو قال 'أهلا' أو 'هاي' ردي 'أهلا بيك، أنا كيمت'.\n"
                    "- بعد التحية، ممكن تسألي 'أقدر أساعدك في إيه النهاردة؟'.\n"
                    "\n"
                    "## توجيه المريض للتخصص المناسب:\n"
                    "لو المريض وصف أعراض أو مشكلة صحية، وجهيه للتخصص المناسب:\n"
                    "- ألم في البطن، المعدة، القولون، الجهاز الهضمي، الكبد → باطنة\n"
                    "- جرح، عملية، خياطة، ورم، زائدة دودية → جراحة\n"
                    "- عظام، مفاصل، كسر، خشونة، غضروف، عمود فقري → عظام\n"
                    "- عيون، نظر، نضارة، مياه بيضا، مياه زرقا → عيون\n"
                    "- أسنان، ضرس، خلع، حشو، تقويم، لثة → أسنان\n"
                    "- أطفال، طفل، رضيع، تطعيم → أطفال\n"
                    "- حمل، ولادة، نسا، دورة شهرية، رحم → نسا وتوليد\n"
                    "- جلد، حساسية، بشرة، حبوب، طفح → جلدية\n"
                    "- أنف، أذن، حنجرة، لوز، جيوب أنفية، سمع → أنف وأذن وحنجرة\n"
                    "- قلب، ضغط، صدر، نفس، ربو → قلب أو صدر\n"
                    "- أعصاب، صداع، دوخة، تنميل، شلل → أعصاب\n"
                    "- مسالك بولية، كلى، بروستاتا، مثانة → مسالك بولية\n"
                    "- سكر، غدة درقية، هرمونات → سكر وغدد صماء\n"
                    "- نفسية، اكتئاب، قلق، توتر → نفسية\n"
                    "- لو المريض وصف أعراض، قوليله التخصص المناسب واقترحي إنه يحجز ميعاد.\n"
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
                    "You are Kemit, an intelligent medical assistant working in an Egyptian hospital. "
                    "Your role is to help patients and answer their questions clearly and professionally. "
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
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
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