# MedRAG Backend

This is the FastAPI backend for the MedRAG (Medical Retrieval-Augmented Generation) system, integrated with your existing React frontend.

## 🏗️ Architecture

```
React Frontend ↔ FastAPI Backend ↔ Weaviate (Hybrid Search) + OpenAI
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start the Server

```bash
# Method 1: Using the run script
python run.py

# Method 2: Using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

### 4. Verify Installation

- API Documentation: <http://localhost:8000/docs>
- Health Check: <http://localhost:8000/health>
- Test the API: <http://localhost:8000/api/chat/health>
  - TTS Health: <http://localhost:8000/api/tts/health>
  - TTS Voices: <http://localhost:8000/api/tts/voices>
  - TTS Audio (example): <http://localhost:8000/api/tts/audio?text=Hello>

## 📋 API Endpoints

### Documents

- `POST /api/documents/upload` - Upload and process documents
- `GET /api/documents/list` - List all uploaded documents
- `GET /api/documents/stats` - Get document statistics
- `DELETE /api/documents/{filename}` - Delete a document

### Chat

- `POST /api/chat/query` - Query the RAG system
- `GET /api/chat/health` - Check chat system health
- `POST /api/chat/test` - Test the chat functionality

### Analytics

- `GET /api/analytics/overview` - Get analytics overview
- `GET /api/analytics/queries` - Get query logs
- `GET /api/analytics/health` - Get system health
- `GET /api/analytics/stats` - Get detailed statistics

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM | Required |
| `ELEVENLABS_API_KEY` | ElevenLabs API key for TTS | Required for TTS |
| `ELEVENLABS_VOICE_ID` | Default ElevenLabs voice ID | Required for TTS |
| `ELEVENLABS_MODEL` | ElevenLabs model | `eleven_flash_v2_5` |
| `WEAVIATE_URL` | Weaviate server URL | `http://localhost:8081` |
| `UPLOAD_DIR` | Document upload directory | `./uploads` |
| `API_HOST` | Server host | `0.0.0.0` |
| `API_PORT` | Server port | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Supported Document Types

- PDF (`.pdf`)
- Word Documents (`.docx`, `.doc`)
- Text Files (`.txt`)
- Markdown (`.md`)

## 🧪 Testing

### Test Document Upload

```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf"
```

### Test Chat Query

```bash
curl -X POST "http://localhost:8000/api/chat/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What medical information is available?"}'
```

## 🔗 Frontend Integration

Your React frontend can now replace the mock functions with real API calls:

### Replace Patient Dashboard Chat

```typescript
// In PatientDashboard.tsx
const sendMessage = async (message: string) => {
  const response = await fetch('http://localhost:8000/api/chat/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: message })
  });
  const data = await response.json();
  return data.answer;
};
```

### Replace Admin Document Upload

```typescript
// In AdminDashboard.tsx
const handleFileUpload = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8000/api/documents/upload', {
    method: 'POST',
    body: formData
  });
  return await response.json();
};
```

## 📊 Features

- ✅ **Document Processing**: Automatic text extraction from PDFs, Word docs, etc.
- ✅ **Hybrid Search**: Advanced search using Weaviate with vector + keyword search (beta=0.3)
- ✅ **AI Chat**: OpenAI-powered question answering
- ✅ **Analytics**: Query logging and system monitoring
- ✅ **CORS Support**: Ready for React frontend integration
- ✅ **Auto Documentation**: Swagger/OpenAPI docs at `/docs`

## 🛠️ Development

### Project Structure

```
backend/
├── app/
│   ├── api/          # API route handlers
│   ├── core/         # Core RAG components
│   ├── models/       # Pydantic models
│   └── main.py       # FastAPI app
├── data/             # Weaviate data (when running locally)
├── uploads/          # Uploaded documents
├── requirements.txt  # Dependencies
└── run.py           # Server startup script
```

### Adding New Features

1. **New API endpoint**: Add to appropriate router in `app/api/`
2. **New data model**: Add to `app/models/schemas.py`
3. **New core functionality**: Add to `app/core/`

## 🚨 Troubleshooting

### Common Issues

1. **"OpenAI API key not configured"**
   - Ensure `OPENAI_API_KEY` is set in your `.env` file

2. **"No documents found for query"**
   - Upload some documents first using `/api/documents/upload`

3. **CORS errors from React**
   - Ensure your React dev server URL is in `CORS_ORIGINS`

4. **Weaviate connection errors**
   - Ensure Weaviate service is running on port 8081
   - Check WEAVIATE_URL environment variable

### Logs

Check the console output for detailed error messages and processing logs.

## 📈 Next Steps

1. Upload some medical documents via the API
2. Test queries through the chat endpoint
3. Update your React frontend to use these APIs
4. Monitor usage through the analytics endpoints

The backend is now ready to power your React frontend with real RAG capabilities!
