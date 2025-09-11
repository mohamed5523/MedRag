import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Users, Shield, Stethoscope, MessageCircle, FileText, BarChart3 } from "lucide-react";

const Index = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light to-background">
      {/* Header */}
      <header className="container mx-auto px-6 py-8">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 bg-gradient-to-br from-primary to-accent rounded-xl flex items-center justify-center">
            <Stethoscope className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-medical-dark">MedRAG System</h1>
        </div>
        <p className="text-muted-foreground max-w-2xl">
          Intelligent hospital management system with RAG-powered document processing and AI assistance
        </p>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 pb-16">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-medical-dark mb-4">Choose Your Role</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Access specialized interfaces designed for your specific needs in the hospital management system
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
          {/* Patient Access */}
          <Card className="p-8 bg-gradient-to-br from-white to-medical-light border-0 shadow-[var(--shadow-card)] hover:shadow-[var(--shadow-hover)] transition-all duration-300 cursor-pointer group"
                onClick={() => navigate('/patient')}>
            <div className="flex flex-col items-center text-center space-y-6">
              <div className="w-20 h-20 bg-gradient-to-br from-primary to-primary-hover rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <Users className="w-10 h-10 text-white" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-medical-dark mb-2">Patient Access</h3>
                <p className="text-muted-foreground mb-6">
                  Chat with our AI assistant to get information about doctors, appointments, and medical services
                </p>
              </div>
              <div className="space-y-3 w-full">
                <div className="flex items-center gap-3 text-sm text-muted-foreground">
                  <MessageCircle className="w-4 h-4 text-primary" />
                  <span>AI-powered chat assistance</span>
                </div>
                <div className="flex items-center gap-3 text-sm text-muted-foreground">
                  <FileText className="w-4 h-4 text-primary" />
                  <span>Doctor information lookup</span>
                </div>
              </div>
              <Button size="lg" className="w-full bg-gradient-to-r from-primary to-primary-hover hover:from-primary-hover hover:to-primary">
                Enter as Patient
              </Button>
            </div>
          </Card>

          {/* Manager Access */}
          <Card className="p-8 bg-gradient-to-br from-white to-accent/5 border-0 shadow-[var(--shadow-card)] hover:shadow-[var(--shadow-hover)] transition-all duration-300 cursor-pointer group"
                onClick={() => navigate('/admin')}>
            <div className="flex flex-col items-center text-center space-y-6">
              <div className="w-20 h-20 bg-gradient-to-br from-accent to-accent-hover rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <Shield className="w-10 h-10 text-white" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-medical-dark mb-2">Hospital Manager</h3>
                <p className="text-muted-foreground mb-6">
                  Access analytics, manage documents, and oversee system usage with comprehensive admin tools
                </p>
              </div>
              <div className="space-y-3 w-full">
                <div className="flex items-center gap-3 text-sm text-muted-foreground">
                  <BarChart3 className="w-4 h-4 text-accent" />
                  <span>Analytics & reporting</span>
                </div>
                <div className="flex items-center gap-3 text-sm text-muted-foreground">
                  <FileText className="w-4 h-4 text-accent" />
                  <span>Document management</span>
                </div>
              </div>
              <Button size="lg" variant="secondary" className="w-full bg-gradient-to-r from-accent to-accent-hover text-white hover:from-accent-hover hover:to-accent">
                Enter as Manager
              </Button>
            </div>
          </Card>
        </div>

        {/* Features Overview */}
        <div className="mt-16 text-center">
          <h3 className="text-2xl font-bold text-medical-dark mb-8">System Features</h3>
          <div className="grid md:grid-cols-3 gap-6 max-w-3xl mx-auto">
            <div className="flex flex-col items-center space-y-3">
              <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center">
                <FileText className="w-6 h-6 text-primary" />
              </div>
              <div>
                <h4 className="font-semibold text-medical-dark">Document Processing</h4>
                <p className="text-sm text-muted-foreground">AI-powered PDF and text analysis</p>
              </div>
            </div>
            <div className="flex flex-col items-center space-y-3">
              <div className="w-12 h-12 bg-accent/10 rounded-xl flex items-center justify-center">
                <MessageCircle className="w-6 h-6 text-accent" />
              </div>
              <div>
                <h4 className="font-semibold text-medical-dark">Smart Chat</h4>
                <p className="text-sm text-muted-foreground">Natural language queries</p>
              </div>
            </div>
            <div className="flex flex-col items-center space-y-3">
              <div className="w-12 h-12 bg-warning/10 rounded-xl flex items-center justify-center">
                <BarChart3 className="w-6 h-6 text-warning" />
              </div>
              <div>
                <h4 className="font-semibold text-medical-dark">Analytics</h4>
                <p className="text-sm text-muted-foreground">Usage insights & reporting</p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;