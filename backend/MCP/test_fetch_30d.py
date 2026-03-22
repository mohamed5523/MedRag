import asyncio
import json
from datetime import datetime, timedelta
from config import get_settings
from http_client import fetch_text

async def main():
    settings = get_settings()
    now = datetime.now()
    d1 = now.strftime("%d/%m/%Y")
    d2 = (now + timedelta(days=30)).strftime("%d/%m/%Y")
    params = {"clinicid": 1, "dateFrom": d1, "dateTo": d2}
    res = await fetch_text(settings.provider_schedule_url, auth=settings.auth, params=params)
    try:
        data = json.loads(res)
        if data.get("data"):
            print(json.dumps(data["data"][:2], ensure_ascii=False, indent=2))
        else:
            print("Empty data:", data)
    except Exception as e:
        print("Raw:", res)

asyncio.run(main())
