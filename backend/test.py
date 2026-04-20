import asyncio
from MCP.clinic_server import match_clinic_hybrid

async def main():
    print(await match_clinic_hybrid('عظام'))

asyncio.run(main())
