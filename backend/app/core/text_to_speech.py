import asyncio
import logging
import os
from typing import Optional

from openai import AsyncOpenAI
from opentelemetry import trace

from .tts_exceptions import TextToSpeechError
from .tts_settings import tts_settings

# ElevenLabs is optional (legacy provider — kept for backward compatibility)
try:
    from elevenlabs import ElevenLabs as _ElevenLabs
except Exception:  # pragma: no cover
    _ElevenLabs = None  # type: ignore

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.tts")

# Azure SDK import
try:
    import azure.cognitiveservices.speech as speechsdk
except Exception as _e:
    speechsdk = None  # Will be validated at runtime


class TextToSpeech:
    """Text-to-speech supporting OpenAI GPT TTS (default), Azure Neural TTS, and ElevenLabs (legacy)."""

    def __init__(self):
        self.provider: str = (tts_settings.TTS_PROVIDER or "openai").lower()
        self._validate_env_vars()
        self._client = None  # ElevenLabs client (legacy), created lazily
        self.openai_client: Optional[AsyncOpenAI] = (
            AsyncOpenAI(api_key=tts_settings.OPENAI_API_KEY)
            if self.provider == "openai"
            else None
        )


    def _validate_env_vars(self) -> None:
        """Validate required environment variables for the selected provider."""
        if self.provider == "openai":
            missing = []
            if not tts_settings.OPENAI_API_KEY:
                missing.append("OPENAI_API_KEY")
            if missing:
                raise ValueError(f"Missing required environment variables for OpenAI TTS: {', '.join(missing)}")
        elif self.provider == "azure":
            missing = []
            if speechsdk is None:
                raise ValueError("azure-cognitiveservices-speech is not installed")
            if not tts_settings.AZURE_TTS_API_KEY:
                missing.append("AZURE_TTS_API_KEY")
            if not tts_settings.AZURE_TTS_ENDPOINT:
                missing.append("AZURE_TTS_ENDPOINT")
            if missing:
                raise ValueError(f"Missing required environment variables for Azure TTS: {', '.join(missing)}")
        elif self.provider == "elevenlabs":
            if _ElevenLabs is None:
                raise ValueError("elevenlabs SDK is not installed. Install it or switch TTS_PROVIDER to 'openai'.")
            missing = [v for v in ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"] if not os.getenv(v)]
            if missing:
                raise ValueError(f"Missing required environment variables for ElevenLabs (legacy): {', '.join(missing)}")
        else:
            raise ValueError("TTS_PROVIDER must be 'openai', 'azure', or 'elevenlabs' (legacy)")

    @property
    def client(self):
        """Get or create ElevenLabs client instance when provider is elevenlabs (legacy)."""
        if self.provider != "elevenlabs":
            return None
        if self._client is None:
            if _ElevenLabs is None:
                raise TextToSpeechError("elevenlabs SDK is not installed")
            self._client = _ElevenLabs(api_key=tts_settings.ELEVENLABS_API_KEY)
        return self._client

    async def synthesize(self, text: str, voice_id: Optional[str] = None, provider: Optional[str] = None) -> bytes:
        """Convert text to speech using the selected provider."""
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        if len(text) > 5000:
            raise ValueError("Input text exceeds maximum length of 5000 characters")

        # Enforce server-side control over provider selection (ignore request override by default)
        selected_provider = (self.provider).lower()

        try:
            with tracer.start_as_current_span("tts.synthesize") as span:
                span.set_attribute("provider", selected_provider)
                span.set_attribute("text.length", len(text))
                if selected_provider == "openai":
                    selected_voice = voice_id or tts_settings.OPENAI_TTS_VOICE
                    span.set_attribute("voice", selected_voice)
                    logger.info(f"[OpenAI TTS] Synthesizing with voice: {selected_voice}")
                    audio_bytes = await self._openai_synthesize(text, selected_voice)
                    if not audio_bytes:
                        raise TextToSpeechError("Generated audio is empty")
                    logger.info(f"[OpenAI TTS] Generated audio: {len(audio_bytes)} bytes")
                    return audio_bytes

                if selected_provider == "azure":
                    selected_voice = voice_id or tts_settings.AZURE_TTS_VOICE
                    span.set_attribute("voice", selected_voice)
                    logger.info(f"[Azure TTS] Synthesizing with voice: {selected_voice}")
                    audio_bytes = await self._azure_synthesize(text, selected_voice)
                    if not audio_bytes:
                        raise TextToSpeechError("Generated audio is empty")
                    logger.info(f"[Azure TTS] Generated audio: {len(audio_bytes)} bytes")
                    return audio_bytes

                if selected_provider == "elevenlabs":
                    selected_voice_id = voice_id or (tts_settings.ELEVENLABS_VOICE_ID or "")
                    span.set_attribute("voice", selected_voice_id)
                    span.set_attribute("model", tts_settings.ELEVENLABS_MODEL)

                    # Normalize text (MSA -> Egyptian dialect) before synthesis
                    from .tts_normalization import normalize_arabic_for_tts
                    normalized_text = normalize_arabic_for_tts(text)
                    logger.info(f"[ElevenLabs] Normalized text: {normalized_text}")

                    logger.info(
                        f"[ElevenLabs] Synthesizing with voice: {selected_voice_id}, "
                        f"model: {tts_settings.ELEVENLABS_MODEL}, "
                        f"stability: {tts_settings.ELEVENLABS_STABILITY}, "
                        f"similarity: {tts_settings.ELEVENLABS_SIMILARITY_BOOST}, "
                        f"style: {tts_settings.ELEVENLABS_STYLE}"
                    )
                    audio_generator = self.client.text_to_speech.convert(
                        text=normalized_text,
                        voice_id=selected_voice_id,
                        model_id=tts_settings.ELEVENLABS_MODEL,
                        output_format=tts_settings.ELEVENLABS_OUTPUT_FORMAT,
                        voice_settings={
                            "stability": tts_settings.ELEVENLABS_STABILITY,
                            "similarity_boost": tts_settings.ELEVENLABS_SIMILARITY_BOOST,
                            "style": tts_settings.ELEVENLABS_STYLE,
                            "use_speaker_boost": True,
                        },
                    )
                    audio_bytes = b"".join(audio_generator)
                    if not audio_bytes:
                        raise TextToSpeechError("Generated audio is empty")
                    logger.info(f"[ElevenLabs] Generated audio: {len(audio_bytes)} bytes")
                    return audio_bytes

            raise ValueError("Unsupported provider. Use 'openai', 'azure' or 'elevenlabs'.")

        except Exception as e:
            logger.error(f"TTS conversion failed ({selected_provider}): {str(e)}")
            raise TextToSpeechError(f"Text-to-speech conversion failed: {str(e)}") from e

    def _azure_output_format_enum(self, fmt_str: str):
        """Map string format to Azure SDK enum."""
        if speechsdk is None:
            return None
        mapping = {
            "audio-24khz-48kbitrate-mono-mp3": speechsdk.SpeechSynthesisOutputFormat.Audio24Khz48KBitRateMonoMp3,
            "audio-24khz-160kbitrate-mono-mp3": speechsdk.SpeechSynthesisOutputFormat.Audio24Khz160KBitRateMonoMp3,
            "audio-16khz-32kbitrate-mono-mp3": speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3,
            "audio-48khz-192kbitrate-mono-mp3": speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3,
            "raw-16khz-16bit-mono-pcm": speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm,
            "raw-24khz-16bit-mono-pcm": speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm,
        }
        return mapping.get((fmt_str or "").lower(), speechsdk.SpeechSynthesisOutputFormat.Audio24Khz48KBitRateMonoMp3)

    async def _azure_synthesize(self, text: str, voice_name: str) -> bytes:
        """Call Azure Neural TTS via SDK in a thread."""
        if speechsdk is None:
            raise TextToSpeechError("azure-cognitiveservices-speech is not installed")

        def _do_speak() -> bytes:
            # Azure Speech SDK requires region, not endpoint
            speech_config = speechsdk.SpeechConfig(
                subscription=tts_settings.AZURE_TTS_API_KEY,
                region=tts_settings.AZURE_TTS_REGION
            )
            speech_config.speech_synthesis_voice_name = voice_name
            fmt_enum = self._azure_output_format_enum(tts_settings.AZURE_TTS_OUTPUT_FORMAT)
            if fmt_enum:
                speech_config.set_speech_synthesis_output_format(fmt_enum)

            # audio_config=None -> capture result.audio_data in memory
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
            result = synthesizer.speak_text_async(text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio = result.audio_data or b""
                return bytes(audio)
            if result.reason == speechsdk.ResultReason.Canceled:
                # Handle cancellation - get details from result properties
                error_details = getattr(result, 'error_details', 'Unknown error')
                cancellation_reason = getattr(result, 'cancellation_details', None)
                if cancellation_reason:
                    error_msg = f"Azure TTS canceled: {cancellation_reason.reason} | {cancellation_reason.error_details or ''}"
                else:
                    error_msg = f"Azure TTS canceled: {error_details}"
                raise TextToSpeechError(error_msg)
            raise TextToSpeechError(f"Azure TTS failed: reason={result.reason}")

        return await asyncio.to_thread(_do_speak)

    async def _openai_synthesize(self, text: str, voice_name: str) -> bytes:
        """Call OpenAI TTS (gpt-4o-mini-tts) using the official Python SDK."""
        # Normalize text first (MSA → Egyptian dialect)
        from .tts_normalization import normalize_arabic_for_tts
        normalized_text = normalize_arabic_for_tts(text)
        logger.info(f"[OpenAI TTS] Normalized text: {normalized_text}")

        # Pronunciation & style guidance for Egyptian dialect TTS
        instructions = (
            "أنت مساعد طبي مصري يتكلم بالعامية المصرية الحضرية (المصري). "
            "اتكلم بطريقة طبيعية وعادية زي أي مصري بيتكلم مع حد قريب منه. "
            "النقاط المهمة للنطق:\n"
            "- الجيم: انطقها جيم مصرية صلبة زي حرف G في كلمة 'go' — مش جيم هشة.\n"
            "- القاف: انطقها زي الهمزة (وقفة في الحلق) في الكلام العادي المصري.\n"
            "- الأسماء الشخصية: انطق أسماء الدكاترة والأشخاص بنطقها الطبيعي المعتاد وبكل وضوح، ولا تغير حروفها.\n"
            "- الأرقام: اقراها بالطريقة المصرية — 'اتنين' مش 'اثنان'، 'تلاتة' مش 'ثلاثة'.\n"
            "- التاريخ والوقت: اقراهم بالطريقة المصرية — 'واحدة و نص' مش 'الساعة الواحدة والنصف'.\n"
            "- العملة: 'جنيه' بنطق مصري طبيعي.\n"
            "- إذا في كلمة أجنبية: انطقها زي ما المصريين بينطقوها.\n"
            "الأسلوب: هادي، ودود، مريح، بيحس بالإنسان. لما بتتكلم عن موضوع طبي، "
            "بطِّى شوية وخلي نبرتك حنينة وواثقة. دايما اتكلم بثقة وحماس للمساعدة."
        )

        try:
            # Use instructions param for gpt-4o-mini-tts (supports it); falls back
            # gracefully for older models like tts-1 / tts-1-hd.
            create_kwargs = dict(
                model=tts_settings.OPENAI_TTS_MODEL,
                voice=voice_name or tts_settings.OPENAI_TTS_VOICE,
                input=normalized_text,
                response_format=tts_settings.OPENAI_TTS_AUDIO_FORMAT,
            )
            if tts_settings.OPENAI_TTS_MODEL and "mini-tts" in tts_settings.OPENAI_TTS_MODEL:
                create_kwargs["instructions"] = instructions

            response = await self.openai_client.audio.speech.create(**create_kwargs)
            audio_bytes = await response.aread()
            return audio_bytes

        except Exception as e:
            raise TextToSpeechError(f"OpenAI TTS failed: {str(e)}")
    def is_available(self) -> bool:
        """Check if TTS service is available for the selected provider."""
        try:
            if self.provider == "openai":
                return bool(tts_settings.OPENAI_API_KEY)
            if self.provider == "azure":
                return bool(tts_settings.AZURE_TTS_API_KEY and tts_settings.AZURE_TTS_ENDPOINT and speechsdk is not None)
            if self.provider == "elevenlabs":
                return self.client is not None
            return False
        except Exception:
            return False

    async def get_available_voices(self) -> list[dict]:
        """Get list of available voices for the current provider."""
        try:
            if self.provider == "azure" and speechsdk is not None:
                def _list():
                    speech_config = speechsdk.SpeechConfig(subscription=tts_settings.AZURE_TTS_API_KEY, endpoint=tts_settings.AZURE_TTS_ENDPOINT)
                    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
                    voices_result = synthesizer.get_voices_async().get()
                    if voices_result.reason == speechsdk.ResultReason.VoicesListRetrieved:
                        vs = []
                        for v in voices_result.voices:
                            vs.append({"id": v.short_name, "name": f"{v.local_name} ({v.locale})", "description": getattr(v, "gender", "")})
                        return vs
                    return []
                voices = await asyncio.to_thread(_list)
                if voices:
                    return voices
                # fallback curated
                return [
                    {"id": tts_settings.AZURE_TTS_VOICE, "name": "Configured Azure Voice", "description": "Default Azure Neural voice"},
                    {"id": "ar-EG-SalmaNeural", "name": "Salma (ar-EG)", "description": "Arabic (Egypt) female"},
                    {"id": "ar-EG-ShakirNeural", "name": "Shakir (ar-EG)", "description": "Arabic (Egypt) male"},
                    # {"id": "ar-SA-HamedNeural", "name": "Hamed (ar-SA)", "description": "Arabic (Saudi Arabia) male"},
                    {"id": "en-US-JennyNeural", "name": "Jenny (en-US)", "description": "US English female"},
                    {"id": "en-US-GuyNeural", "name": "Guy (en-US)", "description": "US English male"},
                ]
            if self.provider == "openai":
                # Curated OpenAI voices (subset). The configured one is included first.
                curated = [
                    {"id": tts_settings.OPENAI_TTS_VOICE, "name": f"Configured OpenAI Voice ({tts_settings.OPENAI_TTS_VOICE})", "description": "Default OpenAI voice"},
                    {"id": "alloy", "name": "Alloy", "description": "Neutral, clear"},
                    {"id": "nova", "name": "Nova", "description": "Friendly, engaging"},
                    {"id": "verse", "name": "Verse", "description": "Natural, warm"},
                    {"id": "shimmer", "name": "Shimmer", "description": "Lively, bright"},
                ]
                # Ensure uniqueness preserving order
                seen = set()
                unique = []
                for v in curated:
                    if v["id"] in seen:
                        continue
                    seen.add(v["id"])
                    unique.append(v)
                return unique
            # ElevenLabs curated list
            return [
                {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam", "description": "Male voice"},
                {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella", "description": "Female voice"},
                {"id": tts_settings.ELEVENLABS_VOICE_ID or "", "name": "Configured Voice", "description": "Currently configured voice"},
            ]
        except Exception:
            return []

    async def health_check(self) -> dict:
        """Check the health status of the TTS service."""
        if not self.is_available():
            return {
                "status": "unhealthy",
                "provider": self.provider,
                "error": "TTS credentials not configured",
            }

        return {
            "status": "healthy",
            "provider": self.provider,
            "available_voices": len(await self.get_available_voices()),
        }