import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ArrowLeft, MessageCircle, Mic, Send, User, Bot, Phone, LogOut, Volume2, VolumeX, RotateCcw } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import { useAuth } from "@/hooks/useAuth";

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  audioData?: string;  // base64 encoded audio
  hasAudio?: boolean;
}

const SESSION_STORAGE_KEY = "medragSessionId";

const PatientDashboard = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { signOut } = useAuth();
  // Build initial greeting based on Eastern European Summer Time (GMT+3)
  const initialGreetingText = (() => {
    const now = new Date();
    const hourEEST = (now.getUTCHours() + 3) % 24; // EEST = UTC+3
    const greeting = hourEEST < 12 ? 'صَبَاح الخِير' : 'مَسَاء الخِير';
    return `${greeting} يا أفندم، مع حضرتك المساعد الشخصي، اسمي كيمت. إزاي أقدر أساعدك النهاردة؟`;
    // return `${greeting} يا أَفَندِم، مَع حَضْرِتِك المُساعِد الشَّخصِي، وَاِسمِي كيمِت، مِن مُستَشفَى مار مَرقُس. مُمكِن أَساعِد حَضْرِتِك إِزَّاي؟`;
  })();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: initialGreetingText,
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [playingAudioId, setPlayingAudioId] = useState<string | null>(null);
  const [canAutoplay, setCanAutoplay] = useState(false);
  const [userGender, setUserGender] = useState<'male' | 'female'>('male');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const generateSessionId = () => {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return `session_${crypto.randomUUID()}`;
    }
    return `session_${Date.now().toString(36)}_${Math.random().toString(36).slice(2)}`;
  };

  const ensureSessionId = (): string | null => {
    if (sessionId) {
      return sessionId;
    }
    if (typeof window === 'undefined') {
      return null;
    }
    let existing = window.localStorage.getItem(SESSION_STORAGE_KEY);
    if (!existing) {
      existing = generateSessionId();
      window.localStorage.setItem(SESSION_STORAGE_KEY, existing);
    }
    setSessionId(existing);
    return existing;
  };

  useEffect(() => {
    if (typeof window === 'undefined') return;
    let existing = window.localStorage.getItem(SESSION_STORAGE_KEY);
    if (!existing) {
      existing = generateSessionId();
      window.localStorage.setItem(SESSION_STORAGE_KEY, existing);
    }
    setSessionId(existing);
  }, []);

  const loadHistory = async (activeSessionId: string) => {
    try {
      const resp = await fetch(`/api/chat/session/history?limit=50`, {
        headers: { "X-Session-Id": activeSessionId }
      });
      if (!resp.ok) return;
      const data = await resp.json();
      const history = Array.isArray(data?.history) ? data.history : [];
      if (!history.length) return;

      const mapped: Message[] = history.map((m: any, idx: number) => ({
        id: `h_${idx}_${Date.now().toString(36)}`,
        text: String(m?.content ?? ""),
        sender: m?.role === "user" ? "user" : "bot",
        timestamp: new Date(((m?.timestamp ?? Date.now() / 1000) as number) * 1000),
        hasAudio: false,
      }));

      // Replace local UI greeting with persisted history if present
      setMessages(mapped);
    } catch {
      // Ignore history load failures; UI can still operate
    }
  };

  useEffect(() => {
    if (!sessionId) return;
    loadHistory(sessionId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const handleNewChat = async () => {
    try {
      const prev = ensureSessionId();
      const resp = await fetch("/api/chat/session/new", {
        method: "POST",
        headers: prev ? { "X-Session-Id": prev } : undefined
      });
      if (resp.ok) {
        const data = await resp.json();
        const newId = String(data?.session_id || "");
        if (newId) {
          window.localStorage.setItem(SESSION_STORAGE_KEY, newId);
          setSessionId(newId);
        } else {
          // fallback: client-generated
          const localNew = generateSessionId();
          window.localStorage.setItem(SESSION_STORAGE_KEY, localNew);
          setSessionId(localNew);
        }
      } else {
        const localNew = generateSessionId();
        window.localStorage.setItem(SESSION_STORAGE_KEY, localNew);
        setSessionId(localNew);
      }
    } catch {
      const localNew = generateSessionId();
      window.localStorage.setItem(SESSION_STORAGE_KEY, localNew);
      setSessionId(localNew);
    }

    // Reset UI state
    stopAudio();
    setPlayingAudioId(null);
    setMessages([
      {
        id: "1",
        text: initialGreetingText,
        sender: "bot",
        timestamp: new Date(),
      },
    ]);
    toast({ title: "New chat started", description: "Your previous chat has been cleared." });
  };

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const initialGreetingPlayedRef = useRef(false);

  // Cross-browser microphone access with fallbacks and helpful errors
  const requestMicStream = async (): Promise<MediaStream> => {
    // Secure contexts requirement: https or localhost
    const isLocalhost = typeof window !== 'undefined' && /^(localhost|127\.0\.0\.1)$/i.test(window.location.hostname);
    if (typeof window !== 'undefined' && !window.isSecureContext && !isLocalhost) {
      throw new Error('Microphone requires a secure context (HTTPS) or localhost');
    }

    const anyNav = navigator as any;
    if (navigator && navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === 'function') {
      return navigator.mediaDevices.getUserMedia({ audio: true });
    }

    const legacyGetUserMedia = anyNav.getUserMedia || anyNav.webkitGetUserMedia || anyNav.mozGetUserMedia || anyNav.msGetUserMedia;
    if (typeof legacyGetUserMedia === 'function') {
      return new Promise<MediaStream>((resolve, reject) => {
        legacyGetUserMedia.call(navigator, { audio: true }, resolve, (err: any) => reject(err));
      });
    }

    throw new Error('getUserMedia is not supported in this browser');
  };

  // Mock responses for fallback when backend is unavailable
  const generateBotResponse = (userMessage: string): string => {
    const message = userMessage.toLowerCase();

    if (message.includes('doctor') || message.includes('physician')) {
      return "I found several doctors in our system. Dr. Sarah Johnson is available for cardiology consultations on weekdays 9-5. Dr. Michael Chen specializes in orthopedics and has openings this Thursday. Would you like more details about any specific doctor?";
    }

    if (message.includes('appointment') || message.includes('schedule')) {
      return "I can help you with appointment information. Currently, we have availability with Dr. Johnson tomorrow at 2:30 PM and Dr. Chen on Friday at 10:00 AM. Which would work better for you?";
    }

    if (message.includes('emergency') || message.includes('urgent')) {
      return "For medical emergencies, please call 911 immediately. For urgent but non-emergency situations, our urgent care is open 24/7 at the main hospital building, Level 2.";
    }

    return "I understand you're asking about '" + userMessage + "'. Based on our hospital database, I'd be happy to help you find more specific information. Could you please provide more details about what you're looking for?";
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputMessage,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const outgoing = inputMessage;
    setInputMessage("");

    try {
      const activeSessionId = ensureSessionId();
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (activeSessionId) {
        headers["X-Session-Id"] = activeSessionId;
      }

      // Use the new endpoint with voice support
      const resp = await fetch("/api/chat/query-with-voice", {
        method: "POST",
        headers,
        body: JSON.stringify({
          query: outgoing,
          max_results: 5,
          user_gender: userGender
        })
      });

      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const data = await resp.json();
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: data.answer ?? generateBotResponse(outgoing),
        sender: 'bot',
        timestamp: new Date(),
        audioData: data.audio_data,
        hasAudio: data.has_audio
      };
      setMessages(prev => [...prev, botMessage]);

      // Auto-play audio if available
      if (data.has_audio && data.audio_data) {
        setTimeout(() => playAudio(botMessage.id, data.audio_data), 500);
      }
    } catch (err) {
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: generateBotResponse(outgoing),
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMessage]);
      toast({
        title: "Using demo responses",
        description: "Backend not reachable; falling back to mock answers.",
      });
    }
  };

  const startRecording = async () => {
    try {
      const stream = await requestMicStream();

      if (typeof (window as any).MediaRecorder === 'undefined') {
        throw new Error('MediaRecorder is not supported in this browser');
      }

      // Prefer webm/opus; some browsers ignore mimeType and choose best available
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        audioChunksRef.current = [];

        try {
          const formData = new FormData();
          formData.append('audio_file', audioBlob, 'recording.webm');

          const resp = await fetch('/api/asr/transcribe', {
            method: 'POST',
            body: formData,
          });

          if (!resp.ok) throw new Error(`ASR HTTP ${resp.status}`);
          const data = await resp.json();
          const text: string = data.transcribed_text ?? '';

          if (text) {
            setInputMessage(text);
            toast({ title: 'Transcribed', description: text });
          } else {
            toast({ title: 'No speech detected', description: 'Try speaking again' });
          }
        } catch (e: any) {
          toast({ title: 'Transcription failed', description: e?.message || 'Unknown error' });
        }
      };

      mediaRecorder.start();
      setIsListening(true);
      toast({ title: 'Voice recording started', description: 'Speak now' });
    } catch (e: any) {
      toast({ title: 'Microphone error', description: e?.message || 'Permission denied or unsupported browser' });
    }
  };

  const stopRecording = () => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop();
      recorder.stream.getTracks().forEach(t => t.stop());
    }
    setIsListening(false);
    toast({ title: 'Voice recording stopped', description: 'Processing your message...' });
  };

  const handleVoiceToggle = () => {
    if (isListening) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  // Audio playback functions
  const playAudio = (messageId: string, audioData: string) => {
    try {
      // Stop any currently playing audio
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }

      // Convert base64 to blob
      const binaryString = atob(audioData);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      const audioBlob = new Blob([bytes], { type: 'audio/mpeg' });
      const audioUrl = URL.createObjectURL(audioBlob);

      // Create and play audio
      const audio = new Audio(audioUrl);
      audioRef.current = audio;
      setPlayingAudioId(messageId);

      audio.onended = () => {
        setPlayingAudioId(null);
        URL.revokeObjectURL(audioUrl);
        audioRef.current = null;
      };

      audio.onerror = () => {
        setPlayingAudioId(null);
        URL.revokeObjectURL(audioUrl);
        audioRef.current = null;
        toast({
          title: "Audio playback failed",
          description: "Unable to play audio response",
        });
      };

      audio.play().catch(err => {
        console.error('Audio playback error:', err);
        setPlayingAudioId(null);
        toast({
          title: "Audio playback failed",
          description: err.message || "Unable to play audio",
        });
      });
    } catch (error: any) {
      console.error('Error preparing audio:', error);
      toast({
        title: "Audio error",
        description: error.message || "Failed to prepare audio",
      });
    }
  };

  const stopAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
      setPlayingAudioId(null);
    }
  };

  const toggleAudio = (messageId: string, audioData: string) => {
    if (playingAudioId === messageId) {
      stopAudio();
    } else {
      playAudio(messageId, audioData);
    }
  };

  // Detect first user interaction to enable autoplay (browser autoplay policy)
  useEffect(() => {
    const enable = () => setCanAutoplay(true);
    window.addEventListener('pointerdown', enable, { once: true } as AddEventListenerOptions);
    window.addEventListener('keydown', enable, { once: true } as AddEventListenerOptions);
    return () => {
      window.removeEventListener('pointerdown', enable);
      window.removeEventListener('keydown', enable);
    };
  }, []);

  // Synthesize and auto-play the initial greeting on mount
  useEffect(() => {
    const synthesizeInitialGreeting = async () => {
      try {
        const resp = await fetch('/api/tts/synthesize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: initialGreetingText })
        });
        if (!resp.ok) return;
        const data = await resp.json();
        if (data?.audio_data) {
          setMessages(prev => prev.map(m => m.id === '1' ? { ...m, audioData: data.audio_data, hasAudio: true } : m));
          if (canAutoplay && !initialGreetingPlayedRef.current) {
            setTimeout(() => {
              playAudio('1', data.audio_data);
              initialGreetingPlayedRef.current = true;
            }, 500);
          }
        }
      } catch {
        // Silently ignore; text-only fallback
      }
    };
    synthesizeInitialGreeting();
  }, [canAutoplay, initialGreetingText]);

  // Auto-play the initial greeting once interaction occurs and audio is ready
  useEffect(() => {
    if (!canAutoplay || initialGreetingPlayedRef.current) return;
    const first = messages.find(m => m.id === '1');
    if (first?.hasAudio && first.audioData) {
      setTimeout(() => {
        playAudio('1', first.audioData!);
        initialGreetingPlayedRef.current = true;
      }, 500);
    }
  }, [canAutoplay, messages]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light to-background">
      {/* Header */}
      <header className="bg-white border-b border-border shadow-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="sm" onClick={() => navigate('/')}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Home
            </Button>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-primary to-accent rounded-lg flex items-center justify-center">
                <MessageCircle className="w-4 h-4 text-white" />
              </div>
              <h1 className="text-xl font-bold text-medical-dark">Patient Assistant</h1>
            </div>
            <div className="ml-auto">
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setUserGender(prev => prev === 'male' ? 'female' : 'male')}
                  className={`border-primary/30 ${userGender === 'female' ? 'bg-pink-50 text-pink-700 border-pink-200' : 'text-medical-dark'}`}
                >
                  <User className={`w-4 h-4 mr-2 ${userGender === 'female' ? 'text-pink-500' : 'text-blue-500'}`} />
                  {userGender === 'male' ? 'Male' : 'Female'}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleNewChat}
                  className="border-primary/30 hover:bg-primary/10 text-medical-dark"
                >
                  <RotateCcw className="w-4 h-4 mr-2 text-primary" />
                  New Chat
                </Button>
                <Button variant="outline" size="sm" onClick={signOut}>
                  <LogOut className="w-4 h-4 mr-2" />
                  Sign Out
                </Button>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Welcome Card */}
          <Card className="p-6 mb-6 bg-gradient-to-r from-primary/5 to-accent/5 border-0">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
              <div>
                <h2 className="text-2xl font-bold text-medical-dark mb-2">Welcome to MedRAG Assistant</h2>
                <p className="text-muted-foreground">
                  Ask me anything about doctors, appointments, medical services, or hospital information
                </p>
              </div>
              <Button
                onClick={() => navigate('/voice')}
                className="bg-gradient-to-r from-accent to-accent-hover hover:from-accent-hover hover:to-accent"
              >
                <Phone className="w-4 h-4 mr-2" />
                Voice Call
              </Button>
            </div>
          </Card>

          {/* Chat Interface */}
          <Card className="h-[600px] flex flex-col bg-white shadow-[var(--shadow-card)]">
            {/* Messages */}
            <ScrollArea className="flex-1 p-6">
              <div className="space-y-4">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex gap-3 ${message.sender === 'user' ? 'flex-row-reverse' : ''}`}
                  >
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${message.sender === 'user'
                      ? 'bg-primary text-white'
                      : 'bg-accent text-white'
                      }`}>
                      {message.sender === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                    </div>
                    <div className={`max-w-[80%] ${message.sender === 'user' ? 'text-right' : ''}`}>
                      <div className={`inline-block p-3 rounded-2xl ${message.sender === 'user'
                        ? 'bg-primary text-white'
                        : 'bg-muted text-foreground'
                        }`}>
                        <p className="text-sm whitespace-pre-wrap">{message.text}</p>
                      </div>
                      <div className="flex items-center gap-2 mt-1">
                        <p className="text-xs text-muted-foreground">
                          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </p>
                        {message.hasAudio && message.audioData && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => toggleAudio(message.id, message.audioData!)}
                            className="h-6 w-6 p-0"
                          >
                            {playingAudioId === message.id ? (
                              <VolumeX className="w-3 h-3 text-accent" />
                            ) : (
                              <Volume2 className="w-3 h-3 text-accent" />
                            )}
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>

            {/* Input */}
            <div className="border-t border-border p-4">
              <div className="flex gap-2">
                <Input
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  placeholder="Ask about doctors, appointments, or medical services..."
                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                  className="flex-1"
                />
                <Button
                  onClick={handleVoiceToggle}
                  variant={isListening ? "destructive" : "secondary"}
                  size="icon"
                >
                  <Mic className={`w-4 h-4 ${isListening ? 'animate-pulse' : ''}`} />
                </Button>
                <Button onClick={handleSendMessage} disabled={!inputMessage.trim()}>
                  <Send className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </Card>

          {/* Quick Actions */}
          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="p-4 cursor-pointer hover:shadow-lg transition-shadow"
              onClick={() => setInputMessage("Show me available cardiologists")}>
              <h3 className="font-semibold text-medical-dark mb-2">Find Specialists</h3>
              <p className="text-sm text-muted-foreground">Search for doctors by specialty</p>
            </Card>
            <Card className="p-4 cursor-pointer hover:shadow-lg transition-shadow"
              onClick={() => setInputMessage("What are the visiting hours?")}>
              <h3 className="font-semibold text-medical-dark mb-2">Hospital Info</h3>
              <p className="text-sm text-muted-foreground">Get hospital policies and hours</p>
            </Card>
            <Card className="p-4 cursor-pointer hover:shadow-lg transition-shadow"
              onClick={() => setInputMessage("How do I schedule an appointment?")}>
              <h3 className="font-semibold text-medical-dark mb-2">Appointments</h3>
              <p className="text-sm text-muted-foreground">Learn about scheduling</p>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PatientDashboard;