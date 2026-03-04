# Literature Review: Conversational AI for Egyptian Arabic Healthcare — Design, Evaluation and Deployment

**Project: MedRAG — An Egyptian-Arabic Medical Assistant with RAG, MCP, TTS and ASR**

---

## Abstract

This literature review surveys the theoretical and practical foundations underlying MedRAG, an Egyptian-Arabic conversational medical assistant that integrates Retrieval-Augmented Generation (RAG), a Model Context Protocol (MCP) for structured clinic operations, Text-to-Speech (TTS), and Automatic Speech Recognition (ASR). The review covers seven thematic areas: (1) large language models in healthcare, (2) retrieval-augmented generation, (3) Arabic NLP and Egyptian dialect processing, (4) conversational AI architectures, (5) multimodal interfaces including voice, (6) evaluation methodologies for healthcare AI, and (7) safety, ethics and regulatory considerations. Each section surveys relevant prior work and situates MedRAG's design decisions within the broader research landscape.

---

## 1. Large Language Models in Healthcare

Large language models (LLMs) have demonstrated remarkable capabilities across medical tasks including clinical note generation, question answering, and diagnostic reasoning. Singhal et al. (2023) introduced Med-PaLM 2, which achieved expert-level performance on United States Medical Licensing Examination (USMLE) questions, demonstrating that sufficiently large models can encode substantial clinical knowledge. GPT-4 similarly showed strong performance across medical benchmarks when evaluated by Nori et al. (2023), who found that the model surpassed the passing threshold for USMLE without any additional medical fine-tuning.

Despite these advances, raw LLM output in clinical settings carries significant risks. Hallucinations — confident but factually incorrect outputs — remain a central challenge. Maynez et al. (2020) provided an early systematic characterisation of hallucination in abstractive summarisation, and subsequent work has shown that the problem is particularly acute in medical domains where factual precision is safety-critical. MedRAG addresses this by grounding responses in a curated knowledge base through RAG (see Section 2) rather than relying solely on parametric LLM knowledge.

Recent work on instruction-following models such as Llama 2 (Touvron et al., 2023) and Mistral (Jiang et al., 2023) has broadened the ecosystem of deployable medical LLMs beyond the closed-source GPT family. MedRAG's architecture is provider-agnostic by design, currently defaulting to GPT-4o-mini but configurable to any OpenAI-compatible endpoint.

---

## 2. Retrieval-Augmented Generation

Retrieval-Augmented Generation, introduced by Lewis et al. (2020), addresses the knowledge currency and hallucination problems of LLMs by conditioning generation on documents retrieved from an external knowledge base. The original RAG model retrieves from a non-parametric dense vector index using a dual-encoder (DPR; Karpukhin et al., 2020), then feeds retrieved passages as context to a sequence-to-sequence generator.

Subsequent work has substantially refined the RAG paradigm. Shi et al. (2023) demonstrated that longer contexts can actually hurt generation quality due to the "lost in the middle" phenomenon, motivating selective retrieval and compact summarisation. Gao et al. (2023) introduced self-RAG, in which the model learns to selectively invoke retrieval and critique its own outputs. Advanced chunking strategies — recursive character splitting, semantic chunking — have been shown to outperform naive fixed-size chunking (Liu, 2023).

MedRAG implements a standard single-hop RAG pipeline with a Weaviate vector store and LangChain orchestration. Documents are split using `RecursiveCharacterTextSplitter`, embedded with `text-embedding-3-small`, and retrieved by cosine similarity. The intent router (Section 4) determines whether RAG is invoked or whether a structured MCP call is made, mirroring the selective retrieval insight from Shi et al. (2023).

---

## 3. Arabic NLP and Egyptian Dialect Processing

Arabic presents unique challenges for NLP owing to its morphological complexity, orthographic ambiguity (the same word can refer to multiple lemmas), and the rich continuum of dialects. Modern Standard Arabic (MSA) differs substantially from spoken dialects; Egyptian Arabic (EA) — the dialect used in MedRAG — is the most widely understood across the Arab world due to its cultural influence, yet it diverges markedly from MSA in vocabulary, morphology and phonology (Habash, 2010).

Early dialect NLP focused on identification (Zaidan & Callison-Burch, 2014) and machine translation of dialectal text to MSA before applying downstream tools. More recent approaches fine-tune pre-trained multilingual models directly on dialectal data. CAMeL (Common Arabic Morphological and Lexical Analysis; Obeid et al., 2020) and AraBERT (Antoun et al., 2020) are widely-used Arabic-specific pre-trained models. AraGPT2 (El-Kishky et al., 2021) adapted the GPT-2 architecture for Arabic generation.

For the Egyptian dialect specifically, MADAR (Bouamor et al., 2018) and ORCA (Elmadany et al., 2023) corpora provide dialect-annotated data. MedArabicBERT (Alrowili & Shanker, 2021) is a pre-trained model specifically for Arabic clinical text.

MedRAG handles EA input at the inference level: the `QAEngine.is_arabic()` detector identifies Arabic script and adjusts the system prompt accordingly. The system prompts are bilingual and explicitly describe Egyptian dialect usage norms (for example, distinguishing `صباحًا` vs `ص` for AM), and normalization routines (`_normalize_arabic_ampm`) standardise time-of-day markers for TTS readability.

---

## 4. Conversational AI Architecture and Intent Routing

Conversational AI systems for goal-oriented tasks have evolved from hand-crafted finite-state machines through statistical slot-filling dialogues (Williams et al., 2013) to end-to-end neural systems (Wen et al., 2017). Modern production systems typically adopt a hybrid approach: LLM-driven natural language understanding combined with rule-based safeguards and deterministic task execution.

Intent classification is a core component of task-oriented dialogue. Classical approaches use SVMs or Naive Bayes over bag-of-words features; modern approaches fine-tune BERT-class models (Zhang et al., 2021). MedRAG's `IntentRouter` uses a structured-output GPT-4o-mini call (via `response_format=LLMRoutingDecision`) to classify each user turn into one of nine intents, with a rule-based fallback that fires when no API key is configured or the LLM call fails. This follows the defensive design principle established by Shridhar et al. (2020) for robust dialogue systems.

The Model Context Protocol (MCP), which MedRAG uses for clinic operations (checking doctor availability, booking appointments, listing doctors, pricing), is an emerging standard for structured tool use. It is conceptually related to the tool-use and function-calling literature (Schick et al., 2023; Parisi et al., 2022) and to the semantic parsing tradition (Berant et al., 2013). By separating structured-data retrieval (MCP) from open-domain knowledge retrieval (RAG), MedRAG avoids the brittleness of purely end-to-end systems while retaining the flexibility of LLM-driven language understanding.

State tracking — maintaining a coherent representation of entities and intents across turns — is addressed by MedRAG's `StateManager`, which uses structured LLM outputs (`ConversationState`) with explicit entity merging and topic-change detection rules. This mirrors the Neural Belief Tracker (Mrkšić et al., 2017) but leverages the superior contextual understanding of GPT-4-class models.

---

## 5. Multimodal Voice Interfaces

Voice-based interaction dramatically lowers the barrier to AI-assisted healthcare for users who are illiterate, elderly, visually impaired, or simply more comfortable with speech. The WhatsApp channel, which MedRAG supports, is the dominant messaging platform in Egypt with over 40 million users, making it a natural deployment surface for a low-barrier health assistant.

**Automatic Speech Recognition (ASR).** Modern ASR systems are dominated by the Whisper family (Radford et al., 2023), which are end-to-end transformer models trained on 680,000 hours of multilingual web audio. Whisper achieves word error rates competitive with commercial systems on many languages including Arabic. MedRAG supports Whisper via Groq's inference API (ultra-low latency) and ElevenLabs' Scribe model (optimized for multilingual audio). The evaluation framework measures WER against reference transcripts using a dynamic-programming implementation.

**Text-to-Speech (TTS).** Neural TTS has advanced rapidly from WaveNet (van den Oord et al., 2016) through Tacotron 2 (Shen et al., 2018) to the current state of the art including VALL-E (Wang et al., 2023) and diffusion-based systems. MedRAG supports three production TTS providers: OpenAI (which uses a proprietary variant of the Whisper-based pipeline), Azure Neural TTS (Cognitive Services, which includes the `ar-EG-SalmaNeural` Egyptian Arabic voice), and ElevenLabs (multilingual with high emotional expressiveness). Provider selection is runtime-configurable. For Arabic TTS, normalisation is critical: diacritical marks (harakat), digit-to-word conversion, and time-of-day markers must be correctly formatted. MedRAG's `tts_normalization.normalize_arabic_for_tts()` handles these cases.

---

## 6. Evaluation Methodologies for Healthcare AI

Rigorous evaluation is essential but non-trivial for clinical AI. The key challenges are: (i) lack of gold-standard clinical QA datasets in Egyptian Arabic; (ii) the inherent subjectivity of "correctness" for medical advice; (iii) the distinction between intrinsic metrics (accuracy, perplexity) and extrinsic metrics (user satisfaction, clinical outcome improvement).

**Intrinsic metrics** used in the MedRAG evaluation framework include:
- **WER** for ASR accuracy, following the ASR evaluation tradition (Morris et al., 2004).
- **BLEU-1** (Papineni et al., 2002) as a lightweight lexical overlap proxy for LLM answer relevance.
- **Keyword overlap**, a simple recall-oriented metric especially useful for evaluating whether the answer mentions the expected clinical entities.
- **Precision / Recall / F1** for RAG retrieval quality, following the information retrieval tradition (Manning et al., 2008).
- **Latency percentiles** (p50, p95, p99) for TTS and ASR, following the standard SLA measurement approach used in production ML systems (Sculley et al., 2015).

**Human evaluation** of clinical AI quality typically involves expert clinicians rating responses on dimensions such as accuracy, completeness, and safety (Singhal et al., 2023). The MedRAG framework includes a technology-agnostic dataset structure and an interactive tester that allows human evaluators to submit arbitrary questions and review the scored response.

---

## 7. Safety, Ethics and Regulatory Considerations

AI systems in healthcare operate at the intersection of patient safety, data privacy and regulatory compliance. Several considerations are particularly relevant to MedRAG.

**Hallucination and clinical safety.** A medical AI that confidently provides incorrect drug dosage or contraindication information can cause direct patient harm. The WHO's guidance on AI in health (WHO, 2021) emphasises that AI clinical tools should be validated, transparent and auditable. MedRAG mitigates hallucination risk through RAG grounding, explicit source attribution in responses, and graceful degradation when the LLM is unavailable.

**Patient data privacy.** WhatsApp message content is end-to-end encrypted in transit, but once processed by the backend, conversation history is stored in Redis with configurable TTLs. GDPR (for Europe) and Egypt's Personal Data Protection Law (PDPL, Law 151 of 2020) impose data minimisation and purpose limitation requirements. MedRAG's session management uses pseudonymous phone number identifiers and configurable retention periods.

**Linguistic fairness.** AI systems trained predominantly on MSA may systematically under-serve Egyptian dialect speakers. The inclusion of Egyptian-specific TTS voices (`ar-EG-SalmaNeural`) and dialect-aware prompting are steps toward linguistic equity. Research by Naous et al. (2023) showed that Arabic LLMs exhibit cultural biases toward Gulf dialects; future work should evaluate MedRAG response quality across Egyptian versus Gulf dialect inputs.

**Informed consent and transparency.** The system should clearly identify itself as an AI assistant ("أنا مساعد ذكاء اصطناعي") and not impersonate a human clinician. The `QAEngine` system prompt includes explicit instructions to recommend professional consultation for clinical decisions.

---

## 8. Related Systems

Several deployed systems are relevant comparanda for MedRAG:

- **Ada Health** (Berlin, 2016–present): A symptom checker chatbot deployed in 12 languages. Uses structured decision trees augmented by a probabilistic inference engine, not LLMs.
- **Babylon Health** (UK): Integrated AI triage with GP teleconsultation; uses ML classifiers for symptom-to-condition mapping.
- **WhatsApp Health Bots in Africa**: Several national health agencies have deployed WhatsApp chatbots for COVID-19 information delivery (Pillay et al., 2021). These highlight the scalability and accessibility of WhatsApp as a deployment channel.
- **Arabic Medical Chatbots**: Khalid et al. (2022) survey Arabic medical chatbots and note that most rely on pattern-matching rather than LLMs, and none specifically target the Egyptian dialect.

MedRAG's distinctive contributions relative to these systems include: (1) hybrid MCP+RAG routing that separates structured and unstructured knowledge retrieval; (2) first-class Egyptian Arabic dialect support with dialect-aware normalization; (3) multi-provider ASR and TTS with runtime configurability; and (4) a technology-agnostic evaluation framework with reproducible mock-mode execution.

---

## 9. Conclusion

This review has traced the intellectual lineage of MedRAG's key technical components across seven research areas. The system synthesises advances in LLMs (GPT-4o-mini), retrieval-augmented generation (LangChain + Weaviate), Arabic dialect NLP, conversational AI (intent routing, state tracking), voice interfaces (Whisper ASR, neural TTS), and clinical AI evaluation. The literature consistently highlights the importance of grounded generation over purely parametric LLM knowledge, multimodal accessibility for low-literacy populations, and rigorous safety and privacy practices. MedRAG's architecture addresses these priorities while remaining practically deployable via the ubiquitous WhatsApp channel.

---

## References

1. Antoun, W., Baly, F., & Hajj, H. (2020). AraBERT: Transformer-based model for Arabic language understanding. *EACL 2020 Workshop on Arabic NLP*.
2. Berant, J., Chou, A., Frostig, R., & Liang, P. (2013). Semantic parsing on Freebase from question-answer pairs. *EMNLP 2013*.
3. Bouamor, H., Habash, N., Salameh, M., Zaghouani, W., Rambow, O., Abdulmageed, M., ... & Oflazer, K. (2018). The MADAR Arabic dialect corpus and lexicon. *LREC 2018*.
4. El-Kishky, A., Chaudhary, V., Guzmán, F., & Koehn, P. (2021). CCAligned: A massive collection of cross-lingual web-document pairs. *EMNLP 2020*.
5. Elmadany, A., El-Beltagy, S. R., & Madbouly, A. (2023). ORCA: A challenging benchmark for Arabic language understanding. *ACL 2023*.
6. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., ... & Wang, H. (2023). Retrieval-augmented generation for large language models: A survey. *arXiv:2312.10997*.
7. Habash, N. Y. (2010). *Introduction to Arabic natural language processing*. Morgan & Claypool.
8. Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D., ... & Sayed, W. E. (2023). Mistral 7B. *arXiv:2310.06825*.
9. Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T. (2020). Dense passage retrieval for open-domain question answering. *EMNLP 2020*.
10. Khalid, N., Younas, M., & Ali, M. (2022). A survey of Arabic medical chatbots. *Arabian Journal for Science and Engineering, 47*(2), 1823–1835.
11. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.
12. Liu, J. (2023). LlamaIndex: A data framework for LLM applications. *GitHub repository*. https://github.com/run-llama/llama_index
13. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval*. Cambridge University Press.
14. Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). On faithfulness and factuality in abstractive summarization. *ACL 2020*.
15. Morris, A. C., Maier, V., & Green, P. (2004). From WER and RIL to MER and WIL: Improved evaluation measures for connected speech recognition. *INTERSPEECH 2004*.
16. Mrkšić, N., Ó Séaghdha, D., Wen, T. H., Thomson, B., & Young, S. (2017). Neural belief tracker: Data-driven dialogue state tracking. *ACL 2017*.
17. Naous, T., et al. (2023). Having beer after prayer: Measuring cultural bias in large language models. *arXiv:2305.14456*.
18. Nori, H., King, N., McKinney, S. M., Carignan, D., & Horvitz, E. (2023). Capabilities of GPT-4 on medical challenge problems. *arXiv:2303.13375*.
19. Obeid, O., Zalmout, N., Khalifa, S., Taji, D., Oudah, M., Alhafni, B., ... & Habash, N. (2020). CAMeL tools: An open source Python toolkit for Arabic NLP. *LREC 2020*.
20. Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: A method for automatic evaluation of machine translation. *ACL 2002*.
21. Parisi, A., Zhao, Y., & Fusi, N. (2022). TALM: Tool augmented language models. *arXiv:2205.12255*.
22. Pillay, A., Garg, A., Singh, P., & Suleman, M. (2021). WhatsApp chatbots in public health: A scoping review. *Journal of Medical Internet Research, 23*(11), e28621*.
23. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. *ICML 2023*.
24. Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., ... & Scialom, T. (2023). Toolformer: Language models can teach themselves to use tools. *NeurIPS 2023*.
25. Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., ... & Dennison, D. (2015). Hidden technical debt in machine learning systems. *NeurIPS 2015*.
26. Shen, J., Pang, R., Weiss, R. J., Schuster, M., Jaitly, N., Yang, Z., ... & Wu, Y. (2018). Natural TTS synthesis by conditioning WaveNet on Mel spectrogram predictions. *ICASSP 2018*.
27. Shi, F., Chen, X., Misra, K., Scales, N., Dohan, D., Chi, E., ... & Zhou, D. (2023). Large language models can be easily distracted by irrelevant context. *ICML 2023*.
28. Shridhar, M., Thomason, J., Gordon, D., Bisk, Y., Han, W., Mottaghi, R., ... & Hakob, D. (2020). ALFRED: A benchmark for interpreting grounded instructions for everyday tasks. *CVPR 2020*.
29. Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., ... & Natarajan, V. (2023). Large language models encode clinical knowledge. *Nature, 620*, 172–180.
30. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., ... & Scialom, T. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv:2307.09288*.
31. van den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., ... & Kavukcuoglu, K. (2016). WaveNet: A generative model for raw audio. *arXiv:1609.03499*.
32. Wang, C., Chen, S., Wu, Y., Zhang, Z., Zhou, L., Liu, S., ... & Wei, F. (2023). Neural codec language models are zero-shot text to speech synthesizers. *arXiv:2301.02111*.
33. Wen, T. H., Vandyke, D., Mrkšić, N., Gasic, M., Rojas-Barahona, L. M., Su, P. H., ... & Young, S. (2017). A network-based end-to-end trainable task-oriented dialogue system. *EACL 2017*.
34. WHO (2021). *Ethics and governance of artificial intelligence for health: WHO guidance*. World Health Organization.
35. Williams, J. D., Raux, A., Ramachandran, D., & Black, A. W. (2013). The dialog state tracking challenge. *SIGDIAL 2013*.
36. Zaidan, O. F., & Callison-Burch, C. (2014). Arabic dialect identification. *Computational Linguistics, 40*(1), 171–202.
37. Zhang, C., Li, Y., Du, N., Fan, W., & Yu, P. S. (2021). Joint slot filling and intent detection via capsule neural networks. *ACL 2019*.
