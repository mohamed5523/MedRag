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
from openai import OpenAI
from opentelemetry import trace

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.qa_engine")

class QAEngine:
    """
    Question-Answering engine using OpenAI GPT models.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"QA Engine initialized with model: {model}")
    
    def _is_arabic_query(self, text: str) -> bool:
        """Detect if the query is primarily in Arabic."""
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        arabic_chars = len(arabic_pattern.findall(text))
        total_chars = len([c for c in text if c.isalpha()])
        
        # Consider it Arabic if more than 50% of alphabetic characters are Arabic
        return total_chars > 0 and (arabic_chars / total_chars) > 0.5
    
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
        else:
            date_hint = f"Today is {now_dt.strftime('%A')}, {now_dt.strftime('%b %d, %Y')}\."

        time_context_message = (
            f"{date_hint} Current timezone: {tz_name}. "
            "Interpret relative dates (e.g., 'tomorrow') relative to this time."
        )

        return {
            "now_dt": now_dt,
            "tz_name": tz_name,
            "is_arabic": is_arabic,
            "date_hint": date_hint,
            "time_context_message": time_context_message,
        }

    def rewrite_query_with_date_hint(
        self, question: str, time_context: Dict[str, Any] | None = None
    ) -> tuple[str, Dict[str, Any]]:
        """Append the computed date hint to the user query."""

        ctx = time_context or self.build_time_context(question)
        date_hint = ctx["date_hint"]
        tz_name = ctx["tz_name"]
        rewritten = (
            f"{question.strip()}\n\n"
            f"Date hint: {date_hint} Current timezone: {tz_name}."
        )
        return rewritten, ctx

    def answer_question(
        self,
        question: str,
        contexts: List[Document],
        now_dt: datetime | None = None,
        time_context: Dict[str, Any] | None = None,
        chat_history: List[Dict[str, str]] | None = None,
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
                span.set_attribute("context.count", len(context_texts))
                
                # Create context string
                context_str = "\n\n".join([f"Document {i+1}: {text}" for i, text in enumerate(context_texts)])
                
                ctx = time_context or self.build_time_context(question, now_dt)
                is_arabic = ctx["is_arabic"]
            
            # Create system prompt for medical context
            if is_arabic:
                system_prompt_openai_tts = (
                    "إنتِ مُساعِدَة طِبِّيَّة ذَكِيَّة اسمِك كيميت، شَغَّالَة في مُسْتَشْفى مِصري. "
                    "دَورِك إنِّك تِسَاعِدِي المَرضَى وتِرُدِّي على أَسْئِلِتْهُم بالعَامِّيَّة المِصريَّة بِطَريقَة وَاضِحَة، سَهْلَة، ومُهَذَّبَة. "
                    "\n"
                    "قَوَاعِد الإِخْرَاج (إِلزَامِيَّة): "
                    "1. مَتِسْتَخْدِمِيش أي رُموز أو تَنْسِيق زي: * _ # [ ] ( ) ` ~ > | - ! … . "
                    "2. مَتِسْتَخْدِمِيش قَوَايِم أو نُقَط في أول السطور. "
                    "3. مَتِكْتِبِيش وَجُوه تَعبِيرِيَّة أو زَخْرَفَة. "
                    "4. اكتُبِي الرَدّ في فَقْرَة واحِدَة مُتَّصِلَة. "
                    "5. بَعْد مَا تِخْلَصِي، رَاجِعِي الرَدّ وتَأَكِّدِي إن مَفيش أي رمز أو اختصار مخالف. "
                    "\n"
                    "اللُّغَة: "
                    "اِسْتَخْدِمِي كَلِمَات مِصريَّة طَبِيعِيَّة زَيّ: "
                    "دُكتور أو دُكتورة بدل طبيب أو طبيبة، "
                    "مِيعاد بدل موعد، "
                    "إزَّيَّك بدل كيف حالك، "
                    "أيوَة بدل نعم، "
                    "ولأ بدل لا. "
                    "\n"
                    "النُطْق: "
                    "خَلِّي كلامِك بنُطْق مِصري طَبِيعِي وسَلِس، يِدِّي إحسَاس بالاهتِمام والرِّقَّة مع المريض. "
                    "اِفْتَكِرِي إنِّك مُمَثِّلَة صَوْتِيَّة أُنثَى، فَاسْتَخْدِمِي ضَمَايِر ومُفرَدات أنثَوِيَّة زَيّ: خايفَة، شايفَة، مُتَأَكِّدَة، قُلتِ، رُحتِ. "
                    "ضِيفِي تَشكِيل خَفِيف فَقَط لَمَّا يِكُون هِيِسَاعِد فِي النُطْق أو يِمْنَع لَبس، "
                    "زَيّ: تِتَواصَلِي، يِقْدَر، يُمْكِن، يُسَاعِد، أُخْدُه، مَفِيش. "
                    "\n"
                    "الأَرْقَام والأَوْقَات والتَّوَارِيخ: "
                    "مُمكِن تِكْتِبِيها بِالأَرْقَام عَادِي (زي 3:30 أو 27/10/2025)، والمُحرِّك هِيِقْرَاها بِصُوت مِصري صَحّ. "
                    "\n"
                    "مَهْمَا يِكُون السُّؤَال، مَتِجَاوِبِيش غِير بِالاعْتِمَاد على المَعْلُومَات المَوْجُودَة فِي المَسْتَنَدَات الطِّبِّيَّة المُتَاحَة (context_str). "
                    "لَو المَعْلُومَة مِش مَوْجُودَة فِي المَسْتَنَدَات، قُولِي بِوُضُوح إنِّك مِش مُتَأَكِّدَة. "
                    "وَلَو السُّؤَال يِحْتَاج تَدَخُّل فَوْرِي، نَبِّهِي المَريض يِتَواصَل مَع دُكتور حالًا."
                    "مَتِفْتِرِضِيش أي علاقَة أو معلومَة مَش مَكتوبَة في المستندات، "
                    "ولو فيه أكتر من تخصص، خُدِي بس المعلُومَات اللي ليها علاقة بالسؤال نفسه."
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
تذكّر استخدام العامية المصرية، والتشكيل على الكلمات المهمة بالطريقة اللي بتوضّح النُطق المِصري.
لو المعلومات ناقصة، قول بوضوح إنك مش متأكد.
"""
            else:
                user_message = f"""
السؤال: {question}

المستندات التي ليها علاقة بالسؤال:
{context_str}

الرجاء تقديم إجابة شاملة بناءً على المعلومات المتاحة فقط.
"""
            
            # Add time context based on Egypt timezone (Africa/Cairo) to ground relative dates
            tz_name = os.getenv("DEFAULT_TZ", "Africa/Cairo")
            if now_dt is None:
                now_dt = datetime.now(ZoneInfo(tz_name)) if ZoneInfo else datetime.now()

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
                "يناير","فبراير","مارس","أبريل","مايو","يونيو",
                "يوليو","أغسطس","سبتمبر","أكتوبر","نوفمبر","ديسمبر",
            ]

            if is_arabic:
                weekday_str = weekday_ar.get(now_dt.strftime("%A"), now_dt.strftime("%A"))
                month_str = months_ar[now_dt.month - 1]
                date_hint = f"النهاردة {weekday_str}، {now_dt.day} {month_str} {now_dt.year}."
            else:
                date_hint = f"Today is {now_dt.strftime('%A')}, {now_dt.strftime('%b %d, %Y')}."

            time_context_message = (
                f"{date_hint} Current timezone: {tz_name}. "
                f"Interpret relative dates (e.g., 'tomorrow') relative to this time."
            )

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
                span.set_attribute("model", self.model)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=500,
                    top_p=0.9
                )
            
            answer = response.choices[0].message.content.strip()
            
            # Get unique sources
            unique_sources = list(set(sources))
            
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