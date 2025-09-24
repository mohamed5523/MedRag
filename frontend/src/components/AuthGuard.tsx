import { useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "@/hooks/useAuth";
import { Skeleton } from "@/components/ui/skeleton";

interface AuthGuardProps {
  children: React.ReactNode;
  allowedRoles?: ('patient' | 'staff' | 'manager')[];
  requireAuth?: boolean;
}

const AuthGuard = ({ children, allowedRoles, requireAuth = true }: AuthGuardProps) => {
  const { user, profile, loading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    if (loading) return; // Wait for auth state to be determined

    // If authentication is required but user is not logged in
    if (requireAuth && !user) {
      navigate('/auth');
      return;
    }

    // If user is logged in but profile is not loaded yet
    if (user && !profile && !loading) {
      // Profile is still loading, wait a bit more
      return;
    }

    // Role-based redirects after successful login
    if (user && profile) {
      // If user is on auth page, redirect based on role
      if (location.pathname === '/auth') {
        switch (profile.role) {
          case 'patient':
            navigate('/patient');
            return;
          case 'staff':
            navigate('/patient'); // Staff also use chat interface
            return;
          case 'manager':
            navigate('/admin');
            return;
          default:
            navigate('/');
            return;
        }
      }

      // Check if user has permission for current route
      if (allowedRoles && !allowedRoles.includes(profile.role)) {
        // Redirect to appropriate dashboard based on role
        switch (profile.role) {
          case 'patient':
            navigate('/patient');
            break;
          case 'staff':
            navigate('/patient');
            break;
          case 'manager':
            navigate('/admin');
            break;
          default:
            navigate('/');
        }
        return;
      }
    }
  }, [user, profile, loading, navigate, location.pathname, allowedRoles, requireAuth]);

  // Show loading state
  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-medical-light to-background flex items-center justify-center">
        <div className="space-y-4 text-center">
          <Skeleton className="h-12 w-12 rounded-lg mx-auto" />
          <Skeleton className="h-4 w-48" />
          <Skeleton className="h-3 w-32" />
        </div>
      </div>
    );
  }

  // If auth is required but user is not logged in, don't render children
  if (requireAuth && !user) {
    return null;
  }

  // If user doesn't have required role, don't render children
  if (allowedRoles && profile && !allowedRoles.includes(profile.role)) {
    return null;
  }

  return <>{children}</>;
};

export default AuthGuard;