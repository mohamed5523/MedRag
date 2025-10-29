# 🚀 Quick Start: TTS Integration

## 📋 Prerequisites

Before testing the TTS feature, ensure you have:

1. **OpenAI API Key** ✅ (default provider)
2. **Environment Variables Set** ✅
3. **Backend Dependencies Installed** ✅
4. (Optional) **ElevenLabs API Key** if you want to use ElevenLabs

## ⚙️ Setup Steps

### 1. Backend Configuration

Create or update `/heal-query-hub/backend/.env`:

```env
# OpenAI Configuration (default TTS provider)
OPENAI_API_KEY=your_openai_key_here
TTS_PROVIDER=openai
OPENAI_TTS_MODEL=gpt-4o-mini-tts
OPENAI_TTS_VOICE=nova
OPENAI_TTS_AUDIO_FORMAT=mp3

# ElevenLabs TTS Configuration (optional)
# ELEVENLABS_API_KEY=your_elevenlabs_key_here
# ELEVENLABS_VOICE_ID=IES4nrmZdUBHByLBde0P
# ELEVENLABS_MODEL=eleven_flash_v2_5
```

### 2. Install Backend Dependencies

```bash
cd heal-query-hub/backend
uv add elevenlabs pydantic-settings
uv sync
```

### 3. Start Backend Server

```bash
cd heal-query-hub/backend
uv run python run.py
```

**Expected output:**

```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     TTS service initialized successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. Start Frontend

```bash
cd heal-query-hub/frontend
npm install  # if first time
npm run dev
```

**Expected output:**

```
VITE v5.x.x  ready in xxx ms

➜  Local:   http://localhost:8080/
```

## 🧪 Test the Integration

### Method 1: Test TTS API Directly

#### Test Health Endpoint

```bash
curl http://localhost:8000/api/tts/health
```

**Expected response:**

```json
{
  "status": "available",
  "provider": "ElevenLabs",
  "available": true,
  "model": "eleven_flash_v2_5",
  "voice_id": "IES4nrmZdUBHByLBde0P"
}
```

#### Test Voices Endpoint

```bash
curl http://localhost:8000/api/tts/voices
```

**Expected response:**

```json
{
  "voices": [
    {
      "id": "IES4nrmZdUBHByLBde0P",
      "name": "Configured Voice",
      "description": "Currently configured voice"
    },
    {
      "id": "pNInz6obpgDQGcFmaJgB",
      "name": "Adam",
      "description": "Male voice"
    }
    // ... more voices
  ]
}
```

#### Test Audio Generation

```bash
curl "http://localhost:8000/api/tts/audio?text=Hello%20world" -o test.mp3
```

**Expected result:** `test.mp3` file created, play it to verify!

#### Test Chat with Voice

```bash
curl -X POST http://localhost:8000/api/chat/query-with-voice \
  -H "Content-Type: application/json" \
  -d '{"query": "What are your services?", "max_results": 5}'
```

**Expected response:**

```json
{
  "answer": "We offer various medical services...",
  "audio_data": "//uQxAAAAAAAAAAAAAAASW5mbw...",
  "has_audio": true,
  "voice_used": "IES4nrmZdUBHByLBde0P",
  "sources": [...],
  "context_count": 3
}
```

### Method 2: Test in Browser

1. **Open Frontend**: <http://localhost:8080>
2. **Sign In** (or create account if needed)
3. **Go to Patient Dashboard**
4. **Send a Message**: Type "Hello" and press Enter
5. **Listen**: Audio should auto-play after ~500ms
6. **Replay**: Click the speaker icon (🔊) to replay

### Expected Behavior

✅ **Message sent** → User message appears on right
✅ **Bot responds** → Bot message appears on left with text
✅ **Audio plays** → Hear the voice response automatically
✅ **Speaker icon** → Shows next to timestamp
✅ **Click speaker** → Replay audio
✅ **Icon changes** → Shows 🔇 while playing, 🔊 when stopped

## 🎛️ Configuration Options

### Change Voice

Update backend `.env`:

```env
# Popular ElevenLabs Voices:
ELEVENLABS_VOICE_ID=pNInz6obpgDQGcFmaJgB  # Adam (male)
ELEVENLABS_VOICE_ID=EXAVITQu4vr4xnSDxMaL  # Bella (female)
ELEVENLABS_VOICE_ID=2EiwWnXFnvU5JabPnv8n  # Clyde (character)
```

### Change Model

```env
# Available models:
ELEVENLABS_MODEL=eleven_flash_v2_5        # Fastest (recommended)
ELEVENLABS_MODEL=eleven_multilingual_v2    # Multi-language
ELEVENLABS_MODEL=eleven_turbo_v2_5        # Fast + quality
```

### Disable Auto-play

Edit `frontend/src/pages/PatientDashboard.tsx`:

Comment out lines 117-120:

```typescript
// if (data.has_audio && data.audio_data) {
//   setTimeout(() => playAudio(botMessage.id, data.audio_data), 500);
// }
```

## 🐛 Troubleshooting

### Backend Issues

#### "TTS service not available"

**Problem**: Backend can't initialize TTS
**Solution**:

```bash
# Check environment variables
cd heal-query-hub/backend
cat .env | grep ELEVENLABS

# Verify API key is valid
curl https://api.elevenlabs.io/v1/voices \
  -H "xi-api-key: YOUR_API_KEY"
```

#### "Module not found: elevenlabs"

**Problem**: Dependencies not installed
**Solution**:

```bash
cd heal-query-hub/backend
uv add elevenlabs pydantic-settings
uv sync
```

#### Port 8000 already in use

**Problem**: Another process using port 8000
**Solution**:

```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### Frontend Issues

#### Audio doesn't play automatically

**Problem**: Browser autoplay policy
**Solution**: User interaction required first. Click the speaker icon manually.

#### "Audio playback failed"

**Problem**: Invalid audio data
**Solution**:

1. Check backend console for TTS errors
2. Test TTS endpoint directly (see above)
3. Verify base64 data in response

#### Network error connecting to backend

**Problem**: Frontend can't reach backend
**Solution**:

```bash
# Check backend is running
curl http://localhost:8000/api/chat/health

# Check CORS settings in backend/app/main.py
```

### Audio Quality Issues

#### Audio sounds robotic

**Problem**: Low-quality model or voice
**Solution**: Try different voice or model in `.env`

#### Audio is too fast/slow

**Problem**: Model speed settings
**Solution**: Currently not configurable, but can be added to backend settings

## 📊 Testing Checklist

Before considering integration complete, verify:

- [ ] Backend starts without errors
- [ ] TTS health endpoint returns "available"
- [ ] Voices endpoint returns list of voices
- [ ] Direct audio generation works (curl test)
- [ ] Chat with voice endpoint works
- [ ] Frontend loads without errors
- [ ] User can send messages
- [ ] Bot responses include text
- [ ] Audio auto-plays for bot responses
- [ ] Speaker icon appears on bot messages
- [ ] Clicking speaker replays audio
- [ ] Icon changes during playback
- [ ] Only one audio plays at a time
- [ ] New audio stops previous audio
- [ ] No console errors in browser

## 🎯 Next Steps

After successful testing:

### 1. **Optimize for Production**

- Add audio caching
- Implement rate limiting
- Add loading indicators
- Handle slow networks

### 2. **Enhance User Experience**

- Add settings to enable/disable auto-play
- Let users choose voice preference
- Add playback speed control
- Show audio duration

### 3. **Monitor Performance**

- Track TTS API usage
- Monitor response times
- Log audio generation errors
- Analyze user engagement with audio

### 4. **WhatsApp Integration** (Your Next Goal!)

- The TTS is now ready
- WhatsApp can send audio messages
- Next: Integrate with WhatsApp Business API
- See WhatsApp credentials section below

## 🔐 WhatsApp Integration Prerequisites

### Required Credentials

1. **WhatsApp Business API Access**
   - Sign up: <https://business.whatsapp.com/>
   - Apply for API access
   - Wait for approval (can take days)

2. **Meta Developer Account**
   - Create app: <https://developers.facebook.com/>
   - Add WhatsApp product
   - Get Phone Number ID
   - Get Access Token

3. **Webhook Setup**
   - Public HTTPS URL required
   - Verify token setup
   - Handle message events
   - Handle status updates

### Environment Variables Needed

```env
# WhatsApp Business API
WHATSAPP_API_TOKEN=your_access_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_id
WHATSAPP_WEBHOOK_VERIFY_TOKEN=your_verify_token
WHATSAPP_BUSINESS_ACCOUNT_ID=your_account_id
```

### Alternative: Twilio WhatsApp

If Meta API is too complex, use Twilio:

```env
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
```

**Docs**: <https://www.twilio.com/docs/whatsapp>

## 📚 Additional Resources

- **ElevenLabs Docs**: <https://docs.elevenlabs.io/>
- **FastAPI Docs**: <https://fastapi.tiangolo.com/>
- **React Audio**: <https://developer.mozilla.org/en-US/docs/Web/API/HTMLAudioElement>
- **WhatsApp Business API**: <https://developers.facebook.com/docs/whatsapp>
- **Twilio WhatsApp**: <https://www.twilio.com/docs/whatsapp/quickstart>

## ✅ Success Criteria

Integration is successful when:

1. ✅ Backend TTS endpoints work
2. ✅ Frontend plays audio automatically
3. ✅ Users can replay audio
4. ✅ No errors in console
5. ✅ Audio quality is good
6. ✅ Ready for WhatsApp integration

**Congratulations! Your MedRAG chatbot now speaks! 🎉🎤**
