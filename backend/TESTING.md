# Testing the Backend Locally

This guide shows you how to test the backend without Docker, before running the full stack.

> 📊 **Want to test with observability enabled?** See [TESTING_WITH_OBSERVABILITY.md](./TESTING_WITH_OBSERVABILITY.md)

## 🚀 Quick Start (Without Phoenix/Observability)

### 1. Disable Observability (Optional)

To test without Phoenix, you can disable observability:

```bash
export ENABLE_OBSERVABILITY=false
```

Or create/update a `.env` file in the backend directory:

```bash
# In backend/.env
ENABLE_OBSERVABILITY=false
```

### 2. Navigate to Backend Directory

```bash
cd heal-query-hub/backend
```

### 3. Install Dependencies

Using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file if you don't have one:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional - for testing without observability
ENABLE_OBSERVABILITY=false

# Optional - server configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true
LOG_LEVEL=info
```

### 5. Run the Backend

```bash
# Method 1: Using the run script
python run.py

# Method 2: Using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

## 🧪 Testing the API

### Manual Testing with Browser

1. **Root endpoint**: http://localhost:8000/
2. **Health check**: http://localhost:8000/health
3. **API Documentation**: http://localhost:8000/docs (Interactive Swagger UI)
4. **Chat health**: http://localhost:8000/api/chat/health
5. **Documents list**: http://localhost:8000/api/documents/list

### Automated Testing Script

Run the provided test script:

```bash
./test_backend.sh
```

Or test a different URL:

```bash
./test_backend.sh http://localhost:8000
```

### Testing with cURL

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Root Endpoint
```bash
curl http://localhost:8000/
```

#### Chat Health
```bash
curl http://localhost:8000/api/chat/health
```

#### Documents List
```bash
curl http://localhost:8000/api/documents/list
```

#### Test Chat Query
```bash
curl -X POST "http://localhost:8000/api/chat/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What medical information is available?"}'
```

## ✅ E2E (Docker Compose) — Real Clinic API (MCP) Regression Tests

This repo includes an end-to-end test script that:
- builds the docker stack
- starts `backend` + `mcp-server` + dependencies
- fetches a real provider list
- runs multi-turn scenarios against `/api/chat/query-with-voice`
  - symptom triage ("اروح لمين؟") should NOT fall back to RAG not-found
  - ambiguous doctor name disambiguation (numbered options) + sticky intent
  - context reset by new session id

### 1) Export required env vars (NO secrets committed)

Set the clinic endpoints + basic auth in your shell:

```bash
export CLINIC_PROVIDER_LIST_URL="..."
export CLINIC_PROVIDER_SCHEDULE_URL="..."
export CLINIC_SERVICE_PRICE_URL="..."
export CLINIC_API_USERNAME="..."
export CLINIC_API_PASSWORD="..."

# backend uses OpenAI to verbalize MCP output
export OPENAI_API_KEY="..."
```

### 2) Run the E2E script

From `heal-query-hub/`:

```bash
./backend/scripts/e2e_real_clinic_api.sh
```

The script uses `docker-compose.e2e.yml` to inject these env vars into the MCP server container
without storing credentials in the repo.

#### Upload a Document
```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/document.pdf"
```

## 🔍 Verify Backend is Running

### Check Logs

When you start the backend, you should see:

```
🏥 Starting MedRAG Backend Server...
📍 Server: http://0.0.0.0:8000
📚 API Docs: http://0.0.0.0:8000/docs
🔧 Environment: Development
```

### Check Health Endpoint

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "medrag-api"
}
```

### Open API Documentation

Open your browser and go to:
```
http://localhost:8000/docs
```

This will show you all available endpoints and let you test them interactively.

## 🐛 Troubleshooting

### Port Already in Use

If port 8000 is already in use:

```bash
# Change the port in .env or use environment variable
export API_PORT=8001
python run.py
```

### Missing Dependencies

```bash
# Install missing packages
pip install -r requirements.txt
# or
uv sync
```

### Observability Connection Errors

If you see Phoenix connection errors but want to test the backend:

```bash
export ENABLE_OBSERVABILITY=false
python run.py
```

### OpenAI API Key Not Set

Make sure your `.env` file contains:

```
OPENAI_API_KEY=sk-your-key-here
```

## ✅ Success Indicators

When the backend is working correctly, you should see:

1. ✅ Server starts without errors
2. ✅ Health endpoint returns `{"status": "healthy"}`
3. ✅ API docs accessible at `/docs`
4. ✅ No connection errors (if observability is disabled)
5. ✅ All endpoints respond (even if empty responses for lists)

## 🚀 Next Steps

Once you've verified the backend works locally:

1. Test the endpoints you'll use in production
2. Upload some test documents
3. Test the chat functionality
4. Test with observability enabled - see [TESTING_WITH_OBSERVABILITY.md](./TESTING_WITH_OBSERVABILITY.md)
5. Then run the full Docker stack with `docker-compose up`

