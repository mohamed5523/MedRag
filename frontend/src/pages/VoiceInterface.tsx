import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft, Mic, MicOff, Phone, PhoneOff, Volume2, LogOut } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import { useAuth } from "@/hooks/useAuth";

const VoiceInterface = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { signOut } = useAuth();
  const [isCallActive, setIsCallActive] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [currentMessage, setCurrentMessage] = useState("");
  const [callDuration, setCallDuration] = useState(0);

  // Mock conversation states
  const mockConversation = [
    { text: "Hello! I'm your AI assistant. How can I help you today?", type: "bot" },
    { text: "I need to know about available cardiologists", type: "user" },
    { text: "I found several cardiologists available. Dr. Sarah Johnson has openings tomorrow at 2:30 PM. Would you like me to provide more details?", type: "bot" },
    { text: "Yes, please tell me more about Dr. Johnson", type: "user" },
    { text: "Dr. Johnson specializes in interventional cardiology with 15 years of experience. She's available Monday through Friday and has excellent patient reviews.", type: "bot" }
  ];

  // Timer for call duration
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isCallActive) {
      interval = setInterval(() => {
        setCallDuration(prev => prev + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isCallActive]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const startCall = () => {
    setIsCallActive(true);
    setCallDuration(0);
    toast({
      title: "Call Started",
      description: "You're now connected to the AI assistant",
    });

    // Start first message
    setTimeout(() => {
      setIsSpeaking(true);
      setCurrentMessage(mockConversation[0].text);
      setTimeout(() => setIsSpeaking(false), 3000);
    }, 1000);
  };

  const endCall = () => {
    setIsCallActive(false);
    setIsListening(false);
    setIsSpeaking(false);
    setCallDuration(0);
    setCurrentMessage("");
    toast({
      title: "Call Ended",
      description: `Call duration: ${formatTime(callDuration)}`,
    });
  };

  const toggleListening = () => {
    if (!isCallActive) return;

    setIsListening(!isListening);

    if (!isListening) {
      toast({
        title: "Listening...",
        description: "Speak now to ask your question",
      });

      // Simulate user speaking and bot response
      setTimeout(() => {
        setIsListening(false);
        setCurrentMessage("Looking up cardiologist information...");

        setTimeout(() => {
          setIsSpeaking(true);
          setCurrentMessage("I found several cardiologists available. Dr. Sarah Johnson has openings tomorrow at 2:30 PM. Would you like more details?");
          setTimeout(() => setIsSpeaking(false), 4000);
        }, 2000);
      }, 3000);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light to-background">
      {/* Header */}
      <header className="bg-white border-b border-border shadow-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="sm" onClick={() => navigate('/patient')}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Chat
            </Button>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-accent to-accent-hover rounded-lg flex items-center justify-center">
                <Phone className="w-4 h-4 text-white" />
              </div>
              <h1 className="text-xl font-bold text-medical-dark">Voice Assistant</h1>
            </div>
            <div className="ml-auto">
              <Button variant="outline" size="sm" onClick={signOut}>
                <LogOut className="w-4 h-4 mr-2" />
                Sign Out
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        <div className="max-w-2xl mx-auto">
          {/* Call Status */}
          <Card className="p-8 text-center mb-8 bg-gradient-to-br from-white to-medical-light shadow-[var(--shadow-card)]">
            <div className="mb-6">
              {isCallActive ? (
                <div className="w-24 h-24 bg-gradient-to-br from-success to-success/80 rounded-full flex items-center justify-center mx-auto mb-4 animate-pulse">
                  <Phone className="w-12 h-12 text-white" />
                </div>
              ) : (
                <div className="w-24 h-24 bg-gradient-to-br from-primary to-accent rounded-full flex items-center justify-center mx-auto mb-4">
                  <Phone className="w-12 h-12 text-white" />
                </div>
              )}

              <h2 className="text-2xl font-bold text-medical-dark mb-2">
                {isCallActive ? "Connected to AI Assistant" : "Ready to Connect"}
              </h2>

              {isCallActive && (
                <div className="text-muted-foreground mb-4">
                  <p className="text-lg font-mono">{formatTime(callDuration)}</p>
                </div>
              )}
            </div>

            {/* Current Status */}
            <div className="mb-6 h-20">
              {isSpeaking && (
                <div className="flex items-center justify-center gap-2 mb-2">
                  <Volume2 className="w-5 h-5 text-accent animate-pulse" />
                  <span className="text-sm font-medium text-accent">AI Speaking...</span>
                </div>
              )}
              {isListening && (
                <div className="flex items-center justify-center gap-2 mb-2">
                  <Mic className="w-5 h-5 text-primary animate-pulse" />
                  <span className="text-sm font-medium text-primary">Listening...</span>
                </div>
              )}
              {currentMessage && (
                <p className="text-sm text-muted-foreground italic px-4 whitespace-pre-wrap">
                  "{currentMessage}"
                </p>
              )}
            </div>

            {/* Call Controls */}
            <div className="flex justify-center gap-4">
              {!isCallActive ? (
                <Button
                  onClick={startCall}
                  size="lg"
                  className="bg-gradient-to-r from-success to-success/80 hover:from-success/90 hover:to-success/70 px-8"
                >
                  <Phone className="w-5 h-5 mr-2" />
                  Start Call
                </Button>
              ) : (
                <>
                  <Button
                    onClick={toggleListening}
                    variant={isListening ? "destructive" : "secondary"}
                    size="lg"
                    disabled={isSpeaking}
                  >
                    {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
                  </Button>
                  <Button
                    onClick={endCall}
                    variant="destructive"
                    size="lg"
                  >
                    <PhoneOff className="w-5 h-5 mr-2" />
                    End Call
                  </Button>
                </>
              )}
            </div>
          </Card>

          {/* Instructions */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold text-medical-dark mb-4">How to Use Voice Assistant</h3>
            <div className="space-y-3 text-sm text-muted-foreground">
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-xs font-bold text-primary">1</span>
                </div>
                <p>Click "Start Call" to connect with the AI assistant</p>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-xs font-bold text-primary">2</span>
                </div>
                <p>Press and hold the microphone button while speaking</p>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-xs font-bold text-primary">3</span>
                </div>
                <p>Ask questions about doctors, appointments, or hospital services</p>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-xs font-bold text-primary">4</span>
                </div>
                <p>Listen to the AI's response and continue the conversation</p>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default VoiceInterface;