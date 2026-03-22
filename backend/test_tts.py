import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv("/home/morad/Projects/heal-query-hub/backend/.env")

async def test_tts():
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("Testing with tts-1...")
    try:
        response = await client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input="مرحبا بك في العيادة.",
            response_format="mp3"
        )
        audio_bytes = await response.aread()
        print(f"Success! Got {len(audio_bytes)} bytes of audio.")
    except Exception as e:
        print(f"Error with tts-1: {e}")
        
    print("Testing with gpt-4o-mini-tts...")
    try:
        response = await client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="nova",
            input="مرحبا بك في العيادة.",
            response_format="mp3"
        )
        audio_bytes = await response.aread()
        print(f"Success! Got {len(audio_bytes)} bytes of audio.")
    except Exception as e:
        print(f"Expected Error with gpt-4o-mini-tts: {e}")

if __name__ == "__main__":
    asyncio.run(test_tts())
