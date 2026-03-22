import asyncio
import json
from config import get_settings
from http_client import fetch_text

async def main():
    settings = get_settings()
    params = {"clinicid": 13, "dateFrom": "10/03/2026", "dateTo": "12/03/2026"}
    res = await fetch_text(settings.provider_schedule_url, auth=settings.auth, params=params)
    try:
        data = json.loads(res)
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        print("Error parsing JSON:", e)
        print("Raw:", res)

asyncio.run(main())
