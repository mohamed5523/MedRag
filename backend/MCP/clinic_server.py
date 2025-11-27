"""Static MCP Server for Clinic Management System."""
from __future__ import annotations

import json
import os
from typing import Optional

from config import DAY_NAME_TO_ID, get_settings
from http_client import fetch_text
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# Create MCP server instance that also powers the HTTP shim
mcp = FastMCP(
    "Clinic Management System",
    host=SERVER_HOST,
    port=SERVER_PORT,
)
settings = get_settings()


@mcp.tool()
async def get_clinic_provider_list() -> str:
    """Get complete list of all clinics and providers.
    
    This endpoint returns all clinics and providers data to use for filtering schedule.
    No filtering parameters are required.
    
    Returns:
        JSON response containing clinic and provider information with Arabic and Latin names
    """
    return await fetch_text(
        settings.provider_list_url,
        auth=settings.auth,
    )


@mcp.tool()
async def get_clinic_provider_schedule(
    clinicid: int,
    dayid: Optional[int] = None,
    providerid: Optional[int] = None,
    day_name: Optional[str] = None
) -> str:
    """Get clinic provider schedule information.
    
    This endpoint returns a list of clinic providers with their related details and schedules.
    You can filter the results using the available query parameters.
    
    Parameters:
        clinicid (int, required): The Clinic ID you need to retrieve data for
        dayid (int, optional): Day ID from 1-7 (1=Saturday, 2=Sunday, 3=Monday, 4=Tuesday, 5=Wednesday, 6=Thursday, 7=Friday)
        providerid (int, optional): Specific provider doctor ID to filter results
        day_name (str, optional): Day name (e.g., 'Monday', 'Saturday') - will be converted to dayid
    
    Returns:
        JSON response containing provider schedule information including shift times
    """
    if day_name and not dayid:
        dayid = DAY_NAME_TO_ID.get(day_name.lower())
    
    # Build request parameters
    params = {}
    params["clinicid"] = clinicid
    if dayid is not None:
        params["dayid"] = dayid
    if providerid is not None:
        params["providerid"] = providerid
    
    return await fetch_text(
        settings.provider_schedule_url,
        auth=settings.auth,
        params=params,
    )


@mcp.tool()
async def get_service_price(
    clinicid: int,
    providerid: Optional[int] = None
) -> str:
    """Get clinic service pricing information.
    
    This endpoint returns service pricing information for a specific clinic and optionally a specific provider.
    
    Parameters:
        clinicid (int, required): The Clinic ID you need to retrieve data for
        providerid (int, optional): Specific provider doctor ID to filter results
    
    Returns:
        JSON response containing service names, prices, doctor information, and clinic details
    """
    # Build request parameters
    params = {}
    params["clinicid"] = clinicid
    if providerid is not None:
        # params["providerid"] = providerid
        params["providerId"] = providerid 
        
    
    return await fetch_text(
        settings.service_price_url,
        auth=settings.auth,
        params=params,
    )


def _json_response_from_text(payload: str) -> Response:
    """Return JSONResponse when possible, otherwise fall back to plain text."""
    try:
        return JSONResponse(json.loads(payload))
    except json.JSONDecodeError:
        return PlainTextResponse(payload, media_type="application/json")


def _parse_int(value: Optional[str], field: str) -> int:
    if value is None:
        raise ValueError(f"{field} is required")
    return int(value)


@mcp.custom_route("/providers", methods=["GET"])
async def providers_route(_request: Request) -> Response:
    data = await get_clinic_provider_list()
    return _json_response_from_text(data)


@mcp.custom_route("/providers/schedule", methods=["GET"])
async def providers_schedule_route(request: Request) -> Response:
    params = request.query_params
    try:
        clinic_id = _parse_int(params.get("clinicid"), "clinicid")
    except (ValueError, TypeError):
        return JSONResponse({"error": "clinicid query parameter is required and must be an integer"}, status_code=400)

    day_id_param = params.get("dayid")
    provider_id_param = params.get("providerid")

    try:
        day_id = int(day_id_param) if day_id_param is not None else None
    except ValueError:
        return JSONResponse({"error": "dayid must be an integer"}, status_code=400)

    try:
        provider_id = int(provider_id_param) if provider_id_param is not None else None
    except ValueError:
        return JSONResponse({"error": "providerid must be an integer"}, status_code=400)

    data = await get_clinic_provider_schedule(
        clinicid=clinic_id,
        dayid=day_id,
        providerid=provider_id,
        day_name=params.get("day_name"),
    )
    return _json_response_from_text(data)


@mcp.custom_route("/providers/services/pricing", methods=["GET"])
async def providers_pricing_route(request: Request) -> Response:
    params = request.query_params
    try:
        clinic_id = _parse_int(params.get("clinicid"), "clinicid")
    except (ValueError, TypeError):
        return JSONResponse({"error": "clinicid query parameter is required and must be an integer"}, status_code=400)

    provider_id_param = params.get("providerid")
    try:
        provider_id = int(provider_id_param) if provider_id_param is not None else None
    except ValueError:
        return JSONResponse({"error": "providerid must be an integer"}, status_code=400)

    data = await get_service_price(
        clinicid=clinic_id,
        providerid=provider_id,
    )
    return _json_response_from_text(data)


# Run the server
if __name__ == "__main__":
    import uvicorn
    # Use the SSE app which includes HTTP routes
    uvicorn.run(
        mcp.sse_app(),
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info"
    )
