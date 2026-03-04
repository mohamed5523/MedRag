# 📊 MedRAG Metrics Guide: A Human-Friendly Explanation

Welcome to the MedRAG Evaluation Guide! This document explains exactly what the dashboard numbers mean. We designed these metrics to give you a clear, objective view of how well the AI hospital assistant is performing—without the technical jargon.

---

## 🤖 1. LLM (Language Model) Quality
*This evaluates how smart, accurate, and fluent the AI is when talking to patients.*

**The Goal:** The AI should answer medical and administrative questions accurately, using natural Egyptian Arabic, without making things up (hallucinating).

### The Metrics:
* **Judge Score (63% weight):** We use a powerful AI (like GPT-4o) as a "Teacher" to grade the assistant's answers out of 1.0. The teacher checks three things:
  * **Factual Correctness (45%):** Did the AI say something medically or factually wrong? (e.g., claiming a doctor is available on a day they aren't).
  * **Relevance (30%):** Did the AI actually answer the patient's question, or did it go off-topic?
  * **Completeness (25%):** Did the AI provide all the necessary details? (e.g., if asked for schedule, did it give the days *and* the hours?).
* **Keyword Rate (27% weight):** Did the AI use the correct Arabic medical vocabulary we expected? If we expect the word "زغللة" (blurred vision) and the AI says "مشكلة في العين" (eye problem), the keyword rate drops. This ensures the AI sounds like a real Egyptian doctor/receptionist.
* **Punctuation Score (10% weight):** This checks if the text is easy to read out loud. If the AI writes a giant block of text without commas (`,`) or periods (`.`), the Text-to-Speech (TTS) voice won't breathe, and it will sound robotic. A high score means the text is well-paced.

---

## 📋 2. MCP (Clinic Operations & Routing)
*This evaluates if the AI successfully connects to your real hospital database to check schedules, prices, and doctor lists.*

**The Goal:** The AI must extract the correct details from the patient's message (like "دكتور إبرام" and "جراحة") and use them to fetch the real data.

### The Metrics:
* **Slot Coverage (45% weight):** "Slots" are the important entities in a sentence (Doctor Name, Clinic Name, Date). Let's say a patient asks: *"متى يتواجد دكتور بيمن في الجراحة؟"*. The AI must detect *"بيمن"* (Doctor) and *"الجراحة"* (Clinic). If the final answer discusses Dr. Bemen and the Surgery clinic, slot coverage is 100%. If it randomly talks about another doctor, it drops.
* **Keyword Retrieval (45% weight):** Does the AI's answer actually sound like a clinic response? Does it use words like "حجز" (booking), "سعر" (price), or "مواعيد" (schedule)? This proves the AI understood the *intent* of the user.
* **Success Rate (10% weight):** Did the hospital database actually respond? If the backend API crashes or times out, this drops. A 100% success rate means the hospital servers are healthy.

---

## 📚 3. RAG (Knowledge Retrieval)
*This evaluates how well the AI reads your uploaded PDFs (like medical guidelines or hospital policies) to answer questions.*

**The Goal:** When the AI doesn't know the answer, it searches your documents. We measure if it finds the *right* paragraph.

### The Metrics:
* **Excerpt Coverage:** We have a list of hidden "golden snippets" (critical medical phrases) that *must* be in the final answer. If the AI synthesizes an answer that completely misses the core medical fact, this score drops. A high score means the AI is successfully reading and quoting your documents.
* **MRR (Mean Reciprocal Rank):** Imagine Googling something. If the right answer is the 1st link, MRR is 1.0 (Perfect!). If it's the 2nd link, MRR is 0.5. If it's the 3rd, it's 0.33. This measures how well the AI ranks the documents it searches through.
* **Precision:** Out of all the paragraphs the AI pulled from the PDFs, how many were actually useful? High precision means the AI isn't reading useless pages, saving time and money.

---

## 🎙️ 4. ASR (Speech-to-Text / Hearing)
*This evaluates how well the AI understands Egyptian patients when they send voice notes.*

**The Goal:** It should accurately transcribe what the patient said, even with heavy accents or medical terms.

### The Metrics:
* **WER (Word Error Rate):** The percentage of words the AI heard wrong. If a patient says 10 words, and the AI misses 2 of them, the WER is 20%. **Lower is better!** For Arabic, anything under 20% (0.20) is usually very good because the AI can still guess the meaning from context.
* **CER (Character Error Rate):** Like WER, but counts individual letters. This is stricter. If a patient says "دكتور" and the AI hears "دكتوه", the WER says the whole word is wrong (1 error), but the CER says only 1 letter is wrong out of 4.

---

## 🔊 5. TTS (Text-to-Speech / Speaking)
*This evaluates how fast and reliable the AI's voice is when replying to patients.*

**The Goal:** The voice reply should be generated almost instantly so the patient doesn't feel like they are waiting.

### The Metrics:
* **Median Latency:** The average time it takes to generate the voice note. If this is `1.2s`, it means half of the voice notes were generated in 1.2 seconds or less.
* **p95 Latency:** This is the "worst-case scenario". It measures the slowest 5% of responses. If p95 is `3.5s`, it means 95% of patients got their voice note fast, but 5% had to wait 3.5 seconds.
* **Success Rate:** Did the provider (like OpenAI or ElevenLabs) successfully generate the audio without crashing?

---

## 🌟 The Composite Score
If you look at the top of the dashboard, you'll see one big number (e.g., 0.85). This is the **Composite Weighted Score**. It combines everything into a single "Health Grade" for your hospital assistant:
* 🤖 **LLM:** 30% (Brain/Thinking is most important)
* 📋 **MCP:** 20% (Clinic Database is critical)
* 📚 **RAG:** 20% (Medical Knowledge is critical)
* 🎙️ **ASR:** 20% (Hearing the patient correctly)
* 🔊 **TTS:** 5% (Voice generation speed)

If this score is above **0.85**, your AI is ready for production and interacting with real patients!
