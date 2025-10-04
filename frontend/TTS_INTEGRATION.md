# 🎵 TTS Frontend Integration Guide

## Overview

The frontend now supports **automatic Text-to-Speech (TTS)** for bot responses. When the chatbot responds, it automatically generates and plays audio using ElevenLabs TTS.

## ✨ Features Implemented

### 1. **Automatic Voice Responses**

- Every bot response includes audio
- Audio is automatically played after the response appears
- Users can replay audio by clicking the speaker icon

### 2. **Audio Controls**

- **Volume2 Icon** (🔊): Click to play audio
- **VolumeX Icon** (🔇): Shows when audio is playing, click to stop
- **Auto-play**: New responses auto-play after 500ms delay

### 3. **Visual Indicators**

- Speaker icon appears next to bot messages with audio
- Icon changes when audio is playing
- Timestamp shows alongside audio controls

## 🎯 How It Works

### API Integration

The frontend now calls the new voice-enabled endpoint:

```typescript
// Old endpoint (text only)
// const resp = await fetch("/api/chat/query", ...)

// New endpoint (text + audio)
const resp = await fetch("/api/chat/query-with-voice", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: outgoing, max_results: 5 })
});

const data = await resp.json();
// data includes:
// - answer: string (text response)
// - audio_data: string (base64 encoded MP3)
// - has_audio: boolean
// - sources: array
// - context_count: number
```

### Audio Playback Flow

1. **Receive Response** → Backend returns JSON with `audio_data` (base64)
2. **Convert to Blob** → Decode base64 to binary, create audio blob
3. **Create Object URL** → Generate temporary URL for audio
4. **Play Audio** → Use HTML5 Audio API
5. **Cleanup** → Revoke object URL when done

### Code Changes

#### 1. **Message Interface** (`PatientDashboard.tsx`)

```typescript
interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  audioData?: string;  // base64 encoded audio
  hasAudio?: boolean;
}
```

#### 2. **State Management**

```typescript
const [playingAudioId, setPlayingAudioId] = useState<string | null>(null);
const audioRef = useRef<HTMLAudioElement | null>(null);
```

#### 3. **Audio Functions**

- `playAudio(messageId, audioData)` - Decode and play audio
- `stopAudio()` - Stop current playback
- `toggleAudio(messageId, audioData)` - Play/pause toggle

#### 4. **UI Component**

```tsx
{message.hasAudio && message.audioData && (
  <Button
    variant="ghost"
    size="sm"
    onClick={() => toggleAudio(message.id, message.audioData!)}
  >
    {playingAudioId === message.id ? (
      <VolumeX className="w-3 h-3 text-accent" />
    ) : (
      <Volume2 className="w-3 h-3 text-accent" />
    )}
  </Button>
)}
```

## 🧪 Testing

### Test the Integration

1. **Start Backend**:

   ```bash
   cd backend
   uv run python run.py
   ```

2. **Start Frontend**:

   ```bash
   cd frontend
   npm run dev
   ```

3. **Test Chat**:
   - Open <http://localhost:8080>
   - Go to Patient Dashboard
   - Send a message
   - Audio should auto-play
   - Click speaker icon to replay

### Expected Behavior

✅ Bot response appears
✅ Audio plays automatically after 500ms
✅ Speaker icon appears next to timestamp
✅ Icon changes to "stop" during playback
✅ Click icon to replay audio
✅ Only one audio plays at a time

## 🎨 UI/UX Features

### Visual Design

- **Minimal UI**: Small speaker icon doesn't clutter the chat
- **Color Coding**: Accent color (medical theme) for audio icons
- **Responsive**: Works on desktop and mobile
- **Accessible**: Clear visual feedback for audio state

### User Experience

- **Auto-play**: Convenience for voice-first users
- **Manual Control**: Users can replay at will
- **Stop on New**: New audio stops previous playback
- **Error Handling**: Graceful fallback with toast notifications

## 🔧 Configuration Options

### Disable Auto-play (Optional)

If you want to disable auto-play:

```typescript
// Remove or comment this section in handleSendMessage
if (data.has_audio && data.audio_data) {
  setTimeout(() => playAudio(botMessage.id, data.audio_data), 500);
}
```

### Change Audio Format

Backend returns MP3 by default. To support other formats, update the blob type:

```typescript
const audioBlob = new Blob([bytes], { type: 'audio/mpeg' });
// or
const audioBlob = new Blob([bytes], { type: 'audio/wav' });
```

### Adjust Auto-play Delay

Change the delay before auto-playing:

```typescript
// Current: 500ms
setTimeout(() => playAudio(botMessage.id, data.audio_data), 500);

// Faster: 200ms
setTimeout(() => playAudio(botMessage.id, data.audio_data), 200);

// No delay: 0ms
setTimeout(() => playAudio(botMessage.id, data.audio_data), 0);
```

## 🐛 Troubleshooting

### Audio Not Playing

**Problem**: Audio doesn't play automatically

**Solutions**:

1. Check browser console for errors
2. Verify backend is returning `has_audio: true`
3. Check browser autoplay policy (some browsers block autoplay)
4. Ensure base64 data is valid

### Base64 Decode Error

**Problem**: `atob()` fails to decode

**Solutions**:

1. Check backend is sending valid base64
2. Verify no whitespace in base64 string
3. Check audio_data is not null/undefined

### Audio Quality Issues

**Problem**: Audio sounds distorted or choppy

**Solutions**:

1. Check ElevenLabs model settings
2. Verify network speed (large responses)
3. Try different voices in backend `.env`

### Memory Leaks

**Problem**: Browser memory increases over time

**Solutions**:

1. Ensure `URL.revokeObjectURL()` is called
2. Clean up audioRef on unmount
3. Add useEffect cleanup:

```typescript
useEffect(() => {
  return () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
  };
}, []);
```

## 📱 Mobile Considerations

### iOS Safari

- Auto-play may be blocked
- User interaction required for first play
- Use toast to prompt user if auto-play fails

### Android Chrome

- Generally works well
- May have bandwidth concerns
- Consider showing loading indicator for large audio

## 🚀 Future Enhancements

### Potential Features

1. **Voice Speed Control**: Add playback speed controls
2. **Download Audio**: Let users download audio responses
3. **Waveform Visualization**: Show audio waveform while playing
4. **Multiple Voices**: Let users select preferred voice
5. **Text Highlighting**: Sync text highlight with audio playback
6. **Queue System**: Queue multiple audio responses
7. **Background Play**: Continue playing while browsing

### Advanced Features

```typescript
// Voice speed control
audio.playbackRate = 1.5; // 1.5x speed

// Download audio
const downloadAudio = (audioData: string, filename: string) => {
  const link = document.createElement('a');
  link.href = `data:audio/mpeg;base64,${audioData}`;
  link.download = filename;
  link.click();
};

// Waveform visualization
const audioContext = new AudioContext();
const analyser = audioContext.createAnalyser();
// ... implement waveform drawing
```

## 📊 Performance Tips

1. **Lazy Load**: Don't decode audio until needed
2. **Cache Control**: Use browser cache for repeated audio
3. **Compression**: Ensure backend uses efficient audio codec
4. **Progressive Enhancement**: Work without audio if TTS fails
5. **Memory Management**: Clean up object URLs promptly

## 🔐 Security Notes

- Base64 audio is safe to store in memory
- Object URLs are temporary and local
- No audio data is persisted to disk
- HTTPS required for microphone access
- Consider rate limiting on backend

## ✅ Summary

The frontend now fully supports:

- ✅ Automatic voice responses
- ✅ Manual audio playback controls
- ✅ Visual feedback for audio state
- ✅ Error handling and fallbacks
- ✅ Mobile-responsive design
- ✅ Clean memory management

**Result**: Users can now hear bot responses automatically, with full control over playback! 🎉
