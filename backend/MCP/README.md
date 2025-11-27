# Clinic MCP Server

A Model Context Protocol (MCP) server for clinic management that provides APIs for retrieving clinic providers, schedules, and service pricing.

## Features

- ✅ Get complete list of all clinics and providers
- ✅ Query provider schedules by clinic, day, and provider
- ✅ Get service pricing information
- ✅ Built with FastMCP and Docker
- ✅ Automatic retry logic with exponential backoff
- ✅ Health checks included

## Setup

### Prerequisites

- Docker and Docker Compose
- Access to the Clinic APIs (configured in `.env`)

### Quick Start

1. **Configure environment variables** (already done - see `.env` file)

2. **Build and run the server:**

```bash
docker-compose up --build -d
```

3. **Check server status:**

```bash
docker-compose ps
docker logs clinic-mcp
```

4. **Stop the server:**

```bash
docker-compose down
```

## API Endpoints

The server runs on `http://localhost:8000` and provides three main endpoints:

### 1. Get Clinic Provider List

Retrieves all clinics and providers data.

```bash
curl http://localhost:8000/providers
```

**Response:** JSON with clinic and provider information including Arabic and Latin names.

---

### 2. Get Provider Schedule

Retrieves clinic provider schedules with optional filtering.

```bash
# Basic query with clinic ID (required)
curl "http://localhost:8000/providers/schedule?clinicid=1"

# Filter by day ID (1=Saturday, 2=Sunday, 3=Monday, etc.)
curl "http://localhost:8000/providers/schedule?clinicid=1&dayid=3"

# Filter by specific provider
curl "http://localhost:8000/providers/schedule?clinicid=1&providerid=123"

# Filter by day name
curl "http://localhost:8000/providers/schedule?clinicid=1&day_name=Monday"

# Combine filters
curl "http://localhost:8000/providers/schedule?clinicid=1&dayid=3&providerid=123"
```

**Query Parameters:**

- `clinicid` (required): The Clinic ID
- `dayid` (optional): Day ID from 1-7 (1=Saturday through 7=Friday)
- `providerid` (optional): Specific provider/doctor ID
- `day_name` (optional): Day name (e.g., 'Monday', 'Saturday')

---

### 3. Get Service Pricing

Retrieves service pricing information for a clinic.

```bash
# Get all services for a clinic
curl "http://localhost:8000/providers/services/pricing?clinicid=1"

# Filter by specific provider
curl "http://localhost:8000/providers/services/pricing?clinicid=1&providerid=123"
```

**Query Parameters:**

- `clinicid` (required): The Clinic ID
- `providerid` (optional): Specific provider/doctor ID

## Configuration

Environment variables can be set in the `.env` file:

```bash
# Clinic API Endpoints
CLINIC_PROVIDER_LIST_URL=http://192.0.0.192:3003/api/clinicProviderlist
CLINIC_PROVIDER_SCHEDULE_URL=http://192.0.0.192:3002/api/clinicProviderschedule/
CLINIC_SERVICE_PRICE_URL=http://192.0.0.192:3005/api/servicePrice

# API Authentication
CLINIC_API_USERNAME=millen
CLINIC_API_PASSWORD=millen@4321

# HTTP Client Settings
REQUEST_TIMEOUT=30
MAX_RETRIES=3
REQUEST_RETRY_BACKOFF=0.5

# Server Settings
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

## MCP Tool Integration

The server also exposes these endpoints as MCP tools that can be used with MCP clients:

- `get_clinic_provider_list()`: Get all clinics and providers
- `get_clinic_provider_schedule(clinicid, dayid?, providerid?, day_name?)`: Get provider schedules
- `get_service_price(clinicid, providerid?)`: Get service pricing

## Testing

### Test with curl

```bash
# Test health check
curl http://localhost:8000/providers

# Test schedule endpoint
curl "http://localhost:8000/providers/schedule?clinicid=1&dayid=1"

# Test pricing endpoint
curl "http://localhost:8000/providers/services/pricing?clinicid=1"
```

### Python Example

```python
import httpx

base_url = "http://localhost:8000"

# Get all providers
response = httpx.get(f"{base_url}/providers")
providers = response.json()

# Get schedule for clinic 1 on Monday
response = httpx.get(
    f"{base_url}/providers/schedule",
    params={"clinicid": 1, "day_name": "Monday"}
)
schedule = response.json()

# Get pricing for clinic 1
response = httpx.get(
    f"{base_url}/providers/services/pricing",
    params={"clinicid": 1}
)
pricing = response.json()
```

## Architecture

```
┌─────────────────┐
│   MCP Client    │
│  (Claude, etc)  │
└────────┬────────┘
         │
         │ HTTP/SSE
         ▼
┌─────────────────────┐
│  FastMCP Server     │
│  (clinic_server.py) │
│                     │
│  - Provider List    │
│  - Schedules        │
│  - Pricing          │
└────────┬────────────┘
         │
         │ HTTP + Auth
         │ + Retry Logic
         ▼
┌─────────────────────┐
│  Clinic Backend     │
│  APIs (External)    │
│                     │
│  - 192.0.0.192:3003 │
│  - 192.0.0.192:3002 │
│  - 192.0.0.192:3005 │
└─────────────────────┘
```

## Troubleshooting

### Container exits immediately

- Check logs: `docker logs clinic-mcp`
- Verify `.env` file exists and is properly formatted

### "Unable to reach Clinic API" errors

- Verify the backend Clinic APIs are running and accessible
- Check network connectivity to `192.0.0.192`
- Verify authentication credentials in `.env`
- Check `docker logs clinic-mcp` for detailed error messages

### Port already in use

- Stop other services on port 8000
- Or change `SERVER_PORT` in `.env` and update `docker-compose.yaml`

## Development

### Local Development (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python clinic_server.py
```

### Rebuild after changes

```bash
docker-compose down
docker-compose up --build -d
```

## License

See project LICENSE file.
