import logging
import os
import re
from datetime import datetime
from typing import List

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

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

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
    
    def answer_question(self, question: str, contexts: List[Document], now_dt: datetime | None = None) -> dict:
        """
        Generate an answer to the question using the provided contexts.
        Returns a dictionary with answer, sources, and metadata.
        """
        if not self.client:
            return {
                "answer": "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.",
                "sources": [],
                "error": "API key not configured"
            }
        
        try:
            # Extract text content from documents
            context_texts = [doc.page_content for doc in contexts]
            sources = [doc.metadata.get("source", "Unknown") for doc in contexts]
            
            # Create context string
            context_str = "\n\n".join([f"Document {i+1}: {text}" for i, text in enumerate(context_texts)])
            
            # Detect if query is in Arabic
            is_arabic = self._is_arabic_query(question)
            
            # Create system prompt for medical context
            if is_arabic:
                system_prompt = (
                    "إنت مُساعِد طِبِّي ذَكِي شَغَّال في مُسْتَشْفى مِصري. "
                    "دَورَك إنَّك تِسَاعِد المَرضَى وتِرُدّ على أَسْئِلَتِهِم بالعَامِّيَّة المِصريَّة بشكل وَاضِح وسَهْل. "
                    "\n"
                    "قَوَاعِد الإِخْرَاج (إِلزَامِيَّة): "
                    "1. مَتِسْتَخْدِمش أي رُموز أو تَنْسِيق زي: * _ # [ ] ( ) ` ~ > | - ! … . "
                    "2. مَتِسْتَخْدِمش قَوَايِم أو نُقَط في أول السطور. "
                    "3. وَسِّع كُلّ الاختصارات زي 'د.'، 'د/'، 'Dr.' إلى 'دُكتور'. "
                    "4. مَتِسْتَخْدِمش 'ص' أو 'م' بعد الوقت. اكتُب الوقت بالكَلام، زَيّ: "
                    "   'من السَّاعَة تِسعَة الصُبح لِحدّ السَّاعَة اِتنين بعد الضُهر'. "
                    "5. مَتِكْرَّرش علامات التَّرقِيم (زي !!! أو ...). "
                    "6. مَتِسْتَخْدِمش وِجُوه تَعبِيرِيَّة أو زَخْرَفَة. "
                    "7. اكتُب النَّص في فَقْرَة واحدة متَّصِلَة. "
                    "8. بَعْد مَا تِخْلَص، راجِع الرَّدّ وتَأَكِّد إِن مَفيش أي رمز أو اختصار مخالف. "
                    "\n"
                    "ضِيف تَشْكِيل بِيُسَاعِد على النُطْق المِصري، خَاصَّة للكَلِمَات اللي ليها أَكْتَر من مَعْنَى أو نُطْق، زي: "
                    "- دَقِيقَة (وَقْت) / دَقيقَة (صِفَة) "
                    "- ساعَة (وَقْت) "
                    "- دَواء، عِلاج، مِيعاد، كَشْف، دُكتور "
                    "- أَفْعَال مِتِكَرَّرَة زي يُمْكِن، يَجِب، نُقَدِّم، تِرُوح، تِجِي "
                    "\n"
                    "كَمَان ضِيف تَشكِيل لِلأَسَامِي والألْقَاب عَشان النُطْق يِبْقَى صَحّ، "
                    "زَيّ: دُكتور أحمَد مَنسُور، سَارة الجَمَّال، خالِد عَبْد الرَّحمَن. "
                    "\n"
                    "الأَرْقَام: "
                    "مَتِكْتِبْهَاش بِأَرْقَام (زي ١ أو 2)، اِكْتِبها بِالكَلام وبنُطْق مِصري: "
                    "واحِد، اِتنين، تَلاتة، أربَعَة، خَمْسَة، سِتَّة، سَبْعَة، تَمَانِيَة، تِسْعَة، عَشَرَة، حَدَاشَر، اِتناشَر. "
                    "وَلَو رَقْم كَبِير اكتُبه زَيّ ما المِصري بيقُول، زَيّ: أربَعَة وعِشْرين. "
                    "\n"
                    "الأَوْقَات: "
                    "مَتِكْتِبْهَاش بِالأَرْقَام، اكتُبها بِنُطْق عَامِّي طَبِيعِي: "
                    "السَّاعَة واحْدَة، اِتنين، تَلاتة ونُصّ، أربَعَة إلَّا رُبْع، خَمْسَة ونُصّ، سِتَّة إلَّا تِلت، سَبْعَة الصُبح، تِمَانِيَة بعد الضُهر، تِسعَة بالليل. "
                    "\n"
                    "أَيّ وَقت مَكْتُوب بِالأَرْقَام زَيّ (3:30 م) حَوِّله لِنُطْق مِصري زَيّ: "
                    "'السَّاعَة تِلاتة ونُصّ المَغرب' أو 'السَّاعَة تِلاتة ونُصّ بعد الضُهر'. "
                    "\n"
                    "الأَيَّام: "
                    "لَو اللُّغَة فُصْحَى حَوِّلها لِلنُطْق المِصري: "
                    "الأحَد، التَنين (الإثنين)، التَلات (الثلاثاء)، الأربَع (الأربعاء)، الخَمِيس، الجُمعة، السَّبْت. "
                    "\n"
                    "اِفْهَم إن التَّشكِيل هِنا تَشكِيل تَوضِيحِي لِلنُطْق المِصري مش فُصْحَى بَحْت. "
                    "مِثَال: قُول 'دَقِيقَة وَاحْدَة' مش 'دَقِيقَةٌ وَاحِدَةٌ'. "
                    "\n"
                    "اِسْتَخْدِم كَلِمَات مِصْرِيَّة زَيّ: "
                    "دُكتور/دُكتورة بدل طبيب/طبيبة، "
                    "مِيعاد بدل موعد، "
                    "إزَّيَّك بدل كيف حالك، "
                    "أيوَة بدل نعم، "
                    "لأ بدل لا. "
                    "\n"
                    "مَهْمَا يِكُون السُّؤَال، مَتِجَاوِبْش غِير بِالاعْتِمَاد على المَعْلُومَات المَوْجُودَة فِي المَسْتَنَدَات الطِّبِّيَّة المُتَاحَة (context_str). "
                    "لَو المَعْلُومَة مِش مَوْجُودَة فِي المَسْتَنَدَات، قُول بوضُوح إنَّك مَش مُتَأَكِّد. "
                    "وَلَو السُّؤَال يِحْتَاج تَدَخُّل فَوْرِي، نَبِّه المَريض يِتْوَاصَل مَع دُكتور حالًا."
                )

            else:
                system_prompt = (
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
Question: {question}

Context from medical documents:
{context_str}

Please provide a comprehensive answer based on the available information.
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

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": time_context_message},
                {"role": "user", "content": user_message}
            ]
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Low temperature for more consistent medical responses
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