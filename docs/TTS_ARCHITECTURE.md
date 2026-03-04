# 🏗️ TTS Integration Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│                  (React + TypeScript Frontend)                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ↓
        ┌───────────────────────────────────────┐
        │     Patient Dashboard Component       │
        │  - Text input                         │
        │  - Voice recording (mic)              │
        │  - Message display                    │
        │  - Audio playback controls            │
        └───────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ↓                               ↓
        ┌───────────────┐             ┌─────────────────┐
        │  ASR Service  │             │  Chat + Voice   │
        │  (Groq)       │             │     Service     │
        └───────────────┘             └─────────────────┘
                                              │
                                              ↓
        ┌──────────────────────────────────────────────────┐
        │          BACKEND API (FastAPI)                   │
        │                                                  │
        │  ┌─────────────────────────────────────────┐   │
        │  │  /api/chat/query-with-voice             │   │
        │  │  - Receives text query                  │   │
        │  │  - Processes with RAG                   │   │
        │  │  - Generates text response              │   │
        │  │  - Generates audio response             │   │
        │  │  - Returns both                         │   │
        │  └─────────────────────────────────────────┘   │
        │                                                  │
        │  ┌─────────────────────────────────────────┐   │
        │  │  /api/tts/* Endpoints                   │   │
        │  │  - /synthesize  (POST)                  │   │
        │  │  - /health      (GET)                   │   │
        │  │  - /voices      (GET)                   │   │
        │  │  - /audio       (GET)                   │   │
        │  └─────────────────────────────────────────┘   │
        │                                                  │
        └──────────────────────────────────────────────────┘
                                │
                ┌───────────────┴──────────────┐
                ↓                              ↓
        ┌───────────────┐            ┌─────────────────┐
        │   RAG Engine  │            │   TTS Service   │
        │  (LangChain)  │            │ (ElevenLabs)    │
        │               │            │                 │
        │  - Chroma DB  │            │  - Text → MP3   │
        │  - OpenAI LLM │            │  - Base64       │
        └───────────────┘            │  - Fast model   │
                                     └─────────────────┘
                                             │
                                             ↓
                                   ┌──────────────────┐
                                   │  ElevenLabs API  │
                                   │  (Cloud Service) │
                                   └──────────────────┘
```

## Component Architecture

### 1. Frontend Layer

```
PatientDashboard.tsx
├── State Management
│   ├── messages: Message[]
│   ├── inputMessage: string
│   ├── isListening: boolean
│   └── playingAudioId: string | null
│
├── Audio Management
│   ├── audioRef: HTMLAudioElement
│   ├── playAudio(id, data)
│   ├── stopAudio()
│   └── toggleAudio(id, data)
│
├── API Integration
│   ├── fetch('/api/chat/query-with-voice')
│   ├── fetch('/api/asr/transcribe')
│   └── Error handling
│
└── UI Components
    ├── Message bubbles
    ├── Audio controls (🔊/🔇)
    ├── Voice recording button
    └── Text input
```

### 2. Backend Layer

```
Backend Architecture
│
├── API Routes (app/api/)
│   ├── chat.py
│   │   ├── /query (text only)
│   │   └── /query-with-voice (text + audio)
│   │
│   └── tts.py
│       ├── /synthesize
│       ├── /health
│       ├── /voices
│       └── /audio
│
├── Core Services (app/core/)
│   ├── text_to_speech.py
│   │   ├── TextToSpeech class
│   │   ├── ElevenLabs client
│   │   └── synthesize() method
│   │
│   ├── tts_settings.py
│   │   └── TTSSettings (Pydantic)
│   │
│   └── tts_exceptions.py
│       └── TextToSpeechError
│
├── Models (app/models/)
│   └── schemas.py
│       ├── TTSRequest
│       ├── TTSResponse
│       ├── ChatResponseWithAudio
│       └── VoiceInfo
│
└── RAG Engine (app/rag/)
    ├── QA Engine
    └── Document processing
```

## Data Flow Diagrams

### A. Chat with Voice Flow

```
User Types Message
        ↓
Frontend sends POST /api/chat/query-with-voice
        ↓
Backend receives query
        ↓
┌───────────────────┐
│  RAG Processing   │
│  1. Retrieve docs │
│  2. Generate text │
└───────────────────┘
        ↓
Text Response: "We offer cardiology services..."
        ↓
┌───────────────────┐
│  TTS Processing   │
│  1. Validate text │
│  2. Call ElevenLabs│
│  3. Get MP3 audio │
│  4. Encode base64 │
└───────────────────┘
        ↓
JSON Response:
{
  "answer": "We offer...",
  "audio_data": "//uQxAAA...",
  "has_audio": true
}
        ↓
Frontend receives response
        ↓
┌─────────────────────────┐
│  1. Display text        │
│  2. Decode base64       │
│  3. Create Blob         │
│  4. Create Object URL   │
│  5. Play audio          │
│  6. Show speaker icon   │
└─────────────────────────┘
```

### B. Audio Playback Flow

```
Bot Message Appears
        ↓
has_audio = true?
        │
        ├─ Yes ─→ Auto-play after 500ms
        │         ├─ Decode base64
        │         ├─ Create Audio element
        │         ├─ Play audio
        │         └─ Show 🔇 icon
        │
        └─ No ──→ Show text only
                  No audio controls


User Clicks Speaker Icon (🔊)
        ↓
Check if already playing
        │
        ├─ Yes ─→ Stop audio
        │         └─ Change to 🔊
        │
        └─ No ──→ Play audio
                  └─ Change to 🔇
```

### C. Error Handling Flow

```
TTS Request
        ↓
Try ElevenLabs API
        │
        ├─ Success ─→ Return audio
        │
        └─ Error ───→ Check error type
                      │
                      ├─ API Key Invalid
                      │  └─ Log + Return text only
                      │
                      ├─ Rate Limit
                      │  └─ Log + Return text only
                      │
                      ├─ Network Error
                      │  └─ Retry once → Return text only
                      │
                      └─ Unknown
                         └─ Log + Return text only

Frontend receives response
        ↓
has_audio = false?
        └─ Show text only (no audio controls)
```

## Module Dependencies

### Backend Dependencies

```
elevenlabs (2.16.0+)
├── httpx (HTTP client)
├── pydantic (data validation)
└── websockets (streaming)

pydantic-settings (2.11.0+)
├── pydantic
└── python-dotenv

fastapi (existing)
└── Used for API routing
```

### Frontend Dependencies

```
React (18+)
└── useState, useRef, useEffect

TypeScript (5+)
└── Type safety

Browser APIs
├── HTMLAudioElement (audio playback)
├── atob() (base64 decoding)
└── Blob/URL APIs (object URLs)
```

## State Management

### Frontend State

```typescript
// Message state
const [messages, setMessages] = useState<Message[]>([...])

// Input state
const [inputMessage, setInputMessage] = useState("")

// Voice state
const [isListening, setIsListening] = useState(false)

// Audio playback state
const [playingAudioId, setPlayingAudioId] = useState<string | null>(null)

// Refs for audio management
const audioRef = useRef<HTMLAudioElement | null>(null)
const mediaRecorderRef = useRef<MediaRecorder | null>(null)
```

### Backend State

```python
# Singleton TTS client (per worker)
class TextToSpeech:
    def __init__(self):
        self._client: Optional[ElevenLabs] = None
    
    @property
    def client(self):
        if self._client is None:
            self._client = ElevenLabs(api_key=...)
        return self._client

# Request state (per request)
- query: str
- max_results: int
- voice_id: Optional[str]

# Response state (per request)
- answer: str
- audio_data: Optional[str]
- has_audio: bool
- sources: List[Dict]
```

## API Contract

### Request Format

```typescript
// Chat with voice request
interface ChatVoiceRequest {
  query: string;          // User's question
  max_results?: number;   // Max RAG results (default: 5)
}

// TTS request
interface TTSRequest {
  text: string;           // Text to convert
  voice_id?: string;      // Optional voice override
}
```

### Response Format

```typescript
// Chat with voice response
interface ChatResponseWithAudio {
  answer: string;               // Text response
  audio_data?: string;          // Base64 MP3 (optional)
  has_audio: boolean;           // Audio available?
  voice_used?: string;          // Voice ID used
  sources: Array<{...}>;        // RAG sources
  context_count: number;        // # of docs used
}

// TTS response
interface TTSResponse {
  success: boolean;
  audio_data?: string;          // Base64 MP3
  audio_size: number;           // Bytes
  voice_used: string;           // Voice ID
  text_length: number;          // Chars
  error?: string;               // Error message
}
```

## Security Architecture

```
┌─────────────────────────────────────┐
│         Security Layers             │
├─────────────────────────────────────┤
│                                     │
│  1. Environment Variables           │
│     - API keys in .env              │
│     - Never in frontend             │
│     - Never in git                  │
│                                     │
│  2. Request Validation              │
│     - Pydantic models               │
│     - Text length limits            │
│     - Input sanitization            │
│                                     │
│  3. Rate Limiting (TODO)            │
│     - Per IP limits                 │
│     - Per user limits               │
│     - Global limits                 │
│                                     │
│  4. Error Handling                  │
│     - No API keys in errors         │
│     - Generic error messages        │
│     - Detailed logging (backend)    │
│                                     │
│  5. CORS Configuration              │
│     - Allowed origins               │
│     - Allowed methods               │
│     - Credentials handling          │
│                                     │
└─────────────────────────────────────┘
```

## Performance Considerations

### Backend Optimizations

```
1. Singleton Pattern
   └─ One ElevenLabs client per worker
   └─ Reuse HTTP connections

2. Async Operations
   └─ TTS runs concurrently with other operations
   └─ Non-blocking I/O

3. Response Streaming (Future)
   └─ Stream audio chunks
   └─ Start playback before complete

4. Caching (Future)
   └─ Cache common phrases
   └─ Redis for distributed cache
```

### Frontend Optimizations

```
1. Lazy Audio Loading
   └─ Don't decode until needed
   └─ Decode on-demand

2. Object URL Cleanup
   └─ Revoke URLs immediately after use
   └─ Prevent memory leaks

3. State Management
   └─ Only one audio instance
   └─ Stop previous on new play

4. Error Boundaries
   └─ Catch render errors
   └─ Graceful degradation
```

## Scalability Considerations

### Horizontal Scaling

```
Load Balancer
      │
      ├─ Backend Server 1 (FastAPI)
      │   └─ ElevenLabs Client
      │
      ├─ Backend Server 2 (FastAPI)
      │   └─ ElevenLabs Client
      │
      └─ Backend Server N (FastAPI)
          └─ ElevenLabs Client

Considerations:
- Each server has own ElevenLabs client
- Stateless design (no shared state)
- Session affinity not required
- Rate limits shared across servers (TODO)
```

### Database Considerations

```
Future Enhancements:
- Store generated audio in S3/Storage
- Cache audio per text + voice combination
- Track usage metrics
- Implement audit logging
```

## Monitoring & Logging

### Metrics to Track

```
Backend Metrics:
├── TTS API calls/second
├── TTS success rate
├── TTS latency (p50, p95, p99)
├── Error rate by type
├── Audio size distribution
└── Voice usage stats

Frontend Metrics:
├── Audio playback success rate
├── Browser autoplay blocks
├── User replay rate
├── Average audio duration
└── Error rate by type
```

### Logging Strategy

```
Backend Logs:
├── INFO: Successful TTS generation
├── WARNING: TTS unavailable (fallback)
├── ERROR: API failures
└── DEBUG: Request/response details

Frontend Logs:
├── console.log: Normal operations
├── console.warn: Recoverable errors
├── console.error: Critical failures
└── Analytics: User interactions
```

## Testing Architecture

### Backend Tests

```
Unit Tests:
├── test_tts_service.py
│   ├── test_initialization
│   ├── test_synthesize
│   ├── test_error_handling
│   └── test_validation

Integration Tests:
├── test_tts_api.py
│   ├── test_health_endpoint
│   ├── test_voices_endpoint
│   ├── test_synthesize_endpoint
│   └── test_chat_with_voice

E2E Tests:
└── test_full_flow.py
    └── test_complete_chat_flow
```

### Frontend Tests

```
Component Tests:
├── PatientDashboard.test.tsx
│   ├── test_message_display
│   ├── test_audio_controls
│   ├── test_playback
│   └── test_error_handling

Integration Tests:
└── test_api_integration.tsx
    ├── test_chat_api_call
    └── test_audio_playback
```

---

## 🎓 Architecture Patterns Used

1. **Singleton Pattern** - ElevenLabs client initialization
2. **Factory Pattern** - Audio object creation
3. **Strategy Pattern** - Error handling strategies
4. **Observer Pattern** - Audio event listeners
5. **Repository Pattern** - TTS service abstraction
6. **Dependency Injection** - Settings management
7. **Clean Architecture** - Separation of concerns

## 📚 References

- FastAPI Best Practices: <https://fastapi.tiangolo.com/tutorial/>
- React Architecture: <https://reactjs.org/docs/thinking-in-react.html>
- ElevenLabs SDK: <https://github.com/elevenlabs/elevenlabs-python>
- Web Audio API: <https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API>

---

**This architecture supports the current implementation and is designed for future enhancements including WhatsApp integration, caching, and advanced audio features.**
