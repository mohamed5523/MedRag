import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ArrowLeft, MessageCircle, Mic, Send, User, Bot, Phone } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

const PatientDashboard = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: "Hello! I'm your AI assistant. I can help you find information about doctors, appointments, and medical services. What would you like to know?",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [isListening, setIsListening] = useState(false);

  // Mock responses for demonstration
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

  const handleSendMessage = () => {
    if (!inputMessage.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputMessage,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage("");

    // Simulate bot response
    setTimeout(() => {
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: generateBotResponse(inputMessage),
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMessage]);
    }, 1000);
  };

  const handleVoiceToggle = () => {
    setIsListening(!isListening);
    toast({
      title: isListening ? "Voice recording stopped" : "Voice recording started",
      description: isListening ? "Processing your message..." : "Speak now to ask your question",
    });
    
    if (!isListening) {
      // Simulate voice input after 3 seconds
      setTimeout(() => {
        setIsListening(false);
        setInputMessage("What doctors are available for cardiology appointments?");
      }, 3000);
    }
  };

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
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                      message.sender === 'user' 
                        ? 'bg-primary text-white' 
                        : 'bg-accent text-white'
                    }`}>
                      {message.sender === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                    </div>
                    <div className={`max-w-[80%] ${message.sender === 'user' ? 'text-right' : ''}`}>
                      <div className={`inline-block p-3 rounded-2xl ${
                        message.sender === 'user'
                          ? 'bg-primary text-white'
                          : 'bg-muted text-foreground'
                      }`}>
                        <p className="text-sm">{message.text}</p>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </p>
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