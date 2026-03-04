import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Index from "./pages/Index";
import AuthPage from "./pages/AuthPage";
import PatientDashboard from "./pages/PatientDashboard";
import AdminDashboard from "./pages/AdminDashboard";
import ManagerDashboard from "./pages/ManagerDashboard";
import VoiceInterface from "./pages/VoiceInterface";
import NotFound from "./pages/NotFound";
import AuthGuard from "./components/AuthGuard";
import { AuthProvider } from "./hooks/useAuth";
import EvaluationDashboard from "./pages/EvaluationDashboard";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <AuthProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Navigate to="/auth" replace />} />
            <Route path="/auth" element={<AuthGuard requireAuth={false}><AuthPage /></AuthGuard>} />
            <Route
              path="/patient"
              element={
                <AuthGuard allowedRoles={['patient', 'staff']}>
                  <PatientDashboard />
                </AuthGuard>
              }
            />
            <Route
              path="/admin"
              element={
                <AuthGuard allowedRoles={['manager']}>
                  <ManagerDashboard />
                </AuthGuard>
              }
            />
            <Route
              path="/voice"
              element={
                <AuthGuard allowedRoles={['patient', 'staff']}>
                  <VoiceInterface />
                </AuthGuard>
              }
            />
            {/* Legacy admin route redirect */}
            <Route
              path="/admin-old"
              element={
                <AuthGuard allowedRoles={['manager']}>
                  <AdminDashboard />
                </AuthGuard>
              }
            />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            {/* Evaluation dashboard — internal tool, no auth guard */}
            <Route path="/evaluation" element={<EvaluationDashboard />} />
            <Route path="*" element={<NotFound />} />

          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
