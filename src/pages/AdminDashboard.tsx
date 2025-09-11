import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, Upload, FileText, BarChart3, Users, MessageSquare, TrendingUp, Clock, Download, Eye } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";

const AdminDashboard = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [uploadProgress, setUploadProgress] = useState(0);

  // Mock data for demonstration
  const stats = {
    totalQueries: 1247,
    activeUsers: 89,
    documentsProcessed: 156,
    avgResponseTime: 2.3
  };

  const recentQueries = [
    { id: 1, user: "Patient #1234", query: "Available cardiologists", timestamp: "2 minutes ago", answered: true },
    { id: 2, user: "Patient #5678", query: "Emergency room wait times", timestamp: "5 minutes ago", answered: true },
    { id: 3, user: "Patient #9012", query: "Dr. Johnson's schedule", timestamp: "8 minutes ago", answered: true },
    { id: 4, user: "Patient #3456", query: "Pediatric appointments", timestamp: "12 minutes ago", answered: false },
  ];

  const documents = [
    { id: 1, name: "Doctor_Profiles_2024.pdf", type: "PDF", size: "2.4 MB", status: "Processed", uploaded: "2024-01-15" },
    { id: 2, name: "Appointment_Schedule.txt", type: "Text", size: "856 KB", status: "Processed", uploaded: "2024-01-14" },
    { id: 3, name: "Medical_Specialties.pdf", type: "PDF", size: "1.8 MB", status: "Processing", uploaded: "2024-01-13" },
    { id: 4, name: "Hospital_Policies.docx", type: "Document", size: "3.2 MB", status: "Failed", uploaded: "2024-01-12" },
  ];

  const handleFileUpload = () => {
    setUploadProgress(0);
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          toast({
            title: "Upload Complete",
            description: "Document has been processed and added to the RAG system",
          });
          return 100;
        }
        return prev + 10;
      });
    }, 200);
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
              <div className="w-8 h-8 bg-gradient-to-br from-accent to-accent-hover rounded-lg flex items-center justify-center">
                <BarChart3 className="w-4 h-4 text-white" />
              </div>
              <h1 className="text-xl font-bold text-medical-dark">Hospital Manager Dashboard</h1>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card className="p-6 bg-gradient-to-br from-primary/5 to-primary/10 border-0">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground mb-1">Total Queries</p>
                <p className="text-2xl font-bold text-primary">{stats.totalQueries}</p>
              </div>
              <MessageSquare className="w-8 h-8 text-primary" />
            </div>
          </Card>
          
          <Card className="p-6 bg-gradient-to-br from-accent/5 to-accent/10 border-0">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground mb-1">Active Users</p>
                <p className="text-2xl font-bold text-accent">{stats.activeUsers}</p>
              </div>
              <Users className="w-8 h-8 text-accent" />
            </div>
          </Card>
          
          <Card className="p-6 bg-gradient-to-br from-success/5 to-success/10 border-0">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground mb-1">Documents</p>
                <p className="text-2xl font-bold text-success">{stats.documentsProcessed}</p>
              </div>
              <FileText className="w-8 h-8 text-success" />
            </div>
          </Card>
          
          <Card className="p-6 bg-gradient-to-br from-warning/5 to-warning/10 border-0">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground mb-1">Avg Response</p>
                <p className="text-2xl font-bold text-warning">{stats.avgResponseTime}s</p>
              </div>
              <Clock className="w-8 h-8 text-warning" />
            </div>
          </Card>
        </div>

        {/* Main Content Tabs */}
        <Tabs defaultValue="analytics" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
            <TabsTrigger value="documents">Documents</TabsTrigger>
            <TabsTrigger value="queries">Query Logs</TabsTrigger>
          </TabsList>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-6">
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-medical-dark mb-4">Usage Analytics</h3>
              <div className="space-y-6">
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm text-muted-foreground">Daily Queries</span>
                    <span className="text-sm font-medium">247 today</span>
                  </div>
                  <Progress value={75} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm text-muted-foreground">Success Rate</span>
                    <span className="text-sm font-medium">94.2%</span>
                  </div>
                  <Progress value={94} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm text-muted-foreground">Document Coverage</span>
                    <span className="text-sm font-medium">87.5%</span>
                  </div>
                  <Progress value={88} className="h-2" />
                </div>
              </div>
            </Card>

            <div className="grid md:grid-cols-2 gap-6">
              <Card className="p-6">
                <h4 className="font-semibold text-medical-dark mb-4">Top Query Categories</h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Doctor Information</span>
                    <Badge variant="secondary">42%</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Appointments</span>
                    <Badge variant="secondary">28%</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Hospital Policies</span>
                    <Badge variant="secondary">18%</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Emergency Info</span>
                    <Badge variant="secondary">12%</Badge>
                  </div>
                </div>
              </Card>

              <Card className="p-6">
                <h4 className="font-semibold text-medical-dark mb-4">Performance Metrics</h4>
                <div className="space-y-4">
                  <div className="flex items-center gap-3">
                    <TrendingUp className="w-4 h-4 text-success" />
                    <span className="text-sm">Response time improved by 15%</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <TrendingUp className="w-4 h-4 text-success" />
                    <span className="text-sm">User satisfaction up 12%</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <TrendingUp className="w-4 h-4 text-primary" />
                    <span className="text-sm">New documents processed: 23</span>
                  </div>
                </div>
              </Card>
            </div>
          </TabsContent>

          {/* Documents Tab */}
          <TabsContent value="documents" className="space-y-6">
            <Card className="p-6">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-lg font-semibold text-medical-dark">Document Management</h3>
                <Button onClick={handleFileUpload} className="bg-gradient-to-r from-primary to-primary-hover">
                  <Upload className="w-4 h-4 mr-2" />
                  Upload Document
                </Button>
              </div>

              {uploadProgress > 0 && uploadProgress < 100 && (
                <div className="mb-6">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm text-muted-foreground">Uploading...</span>
                    <span className="text-sm font-medium">{uploadProgress}%</span>
                  </div>
                  <Progress value={uploadProgress} className="h-2" />
                </div>
              )}

              <div className="space-y-4">
                {documents.map((doc) => (
                  <div key={doc.id} className="flex items-center justify-between p-4 border border-border rounded-lg">
                    <div className="flex items-center gap-3">
                      <FileText className="w-5 h-5 text-muted-foreground" />
                      <div>
                        <p className="font-medium text-medical-dark">{doc.name}</p>
                        <p className="text-sm text-muted-foreground">{doc.size} • Uploaded {doc.uploaded}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <Badge variant={
                        doc.status === 'Processed' ? 'default' :
                        doc.status === 'Processing' ? 'secondary' : 'destructive'
                      }>
                        {doc.status}
                      </Badge>
                      <Button variant="ghost" size="sm">
                        <Eye className="w-4 h-4" />
                      </Button>
                      <Button variant="ghost" size="sm">
                        <Download className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </TabsContent>

          {/* Query Logs Tab */}
          <TabsContent value="queries" className="space-y-6">
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-medical-dark mb-6">Recent Query Logs</h3>
              <div className="space-y-4">
                {recentQueries.map((query) => (
                  <div key={query.id} className="flex items-center justify-between p-4 border border-border rounded-lg">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-1">
                        <span className="font-medium text-medical-dark">{query.user}</span>
                        <Badge variant={query.answered ? "default" : "secondary"}>
                          {query.answered ? "Answered" : "Pending"}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mb-1">{query.query}</p>
                      <p className="text-xs text-muted-foreground">{query.timestamp}</p>
                    </div>
                    <Button variant="ghost" size="sm">
                      <Eye className="w-4 h-4" />
                    </Button>
                  </div>
                ))}
              </div>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default AdminDashboard;