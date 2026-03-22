import asyncio
from clinic_server import get_clinic_provider_schedule
from datetime import datetime, timedelta

async def main():
    now = datetime.now()
    d1 = now.strftime("%d/%m/%Y")
    d2 = (now + timedelta(days=7)).strftime("%d/%m/%Y")
    # Call the tool directly
    markdown = await get_clinic_provider_schedule(clinicid=13, dateFrom=d1, dateTo=d2)
    print("OUTPUT IS MARKDOWN:")
    print(markdown[:1000])

asyncio.run(main())
