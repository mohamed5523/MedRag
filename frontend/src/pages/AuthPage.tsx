import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ArrowLeft, User, Users, Shield, Stethoscope } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";

const AuthPage = () => {
  const navigate = useNavigate();
  const { signIn, signUp, loading } = useAuth();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [selectedRole, setSelectedRole] = useState<'patient' | 'staff' | 'manager'>('patient');

  // Login form state
  const [loginEmail, setLoginEmail] = useState("");
  const [loginPassword, setLoginPassword] = useState("");

  // Signup form state
  const [signupEmail, setSignupEmail] = useState("");
  const [signupPassword, setSignupPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [name, setName] = useState("");

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!loginEmail || !loginPassword) return;

    setIsSubmitting(true);
    const { error } = await signIn(loginEmail, loginPassword);
    if (!error) {
      // Redirect based on role will be handled by the AuthGuard
      navigate('/');
    }
    setIsSubmitting(false);
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!signupEmail || !signupPassword || !confirmPassword || !name) return;

    if (signupPassword !== confirmPassword) {
      // Handle password mismatch
      return;
    }

    setIsSubmitting(true);
    const { error } = await signUp(signupEmail, signupPassword, selectedRole);
    if (!error) {
      navigate('/');
    }
    setIsSubmitting(false);
  };

  const roleIcons = {
    patient: User,
    staff: Stethoscope,
    manager: Shield
  };

  const roleDescriptions = {
    patient: "Book appointments and chat with our AI assistant",
    staff: "Access patient information and assist with medical inquiries",
    manager: "Manage documents, view analytics, and oversee operations"
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light to-background flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Header */}
        <div className="flex items-center gap-4 mb-6">
          <Button variant="ghost" size="sm" onClick={() => navigate('/')}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Home
          </Button>
        </div>

        <Card className="shadow-[var(--shadow-card)]">
          <CardHeader className="text-center">
            <div className="w-12 h-12 bg-gradient-to-br from-primary to-accent rounded-lg flex items-center justify-center mx-auto mb-3">
              <Users className="w-6 h-6 text-white" />
            </div>
            <CardTitle className="text-2xl font-bold text-medical-dark">MedRAG Assistant</CardTitle>
            <CardDescription>
              Access your personalized healthcare assistant
            </CardDescription>
          </CardHeader>

          <CardContent>
            <Tabs defaultValue="login" className="w-full">
              <TabsList className="grid w-full grid-cols-2 mb-6">
                <TabsTrigger value="login">Sign In</TabsTrigger>
                <TabsTrigger value="signup">Sign Up</TabsTrigger>
              </TabsList>

              <TabsContent value="login">
                <form onSubmit={handleLogin} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="login-email">Email</Label>
                    <Input
                      id="login-email"
                      type="email"
                      value={loginEmail}
                      onChange={(e) => setLoginEmail(e.target.value)}
                      placeholder="Enter your email"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="login-password">Password</Label>
                    <Input
                      id="login-password"
                      type="password"
                      value={loginPassword}
                      onChange={(e) => setLoginPassword(e.target.value)}
                      placeholder="Enter your password"
                      required
                    />
                  </div>
                  <Button 
                    type="submit" 
                    className="w-full bg-gradient-to-r from-primary to-accent hover:from-primary-hover hover:to-accent-hover"
                    disabled={isSubmitting || loading}
                  >
                    {isSubmitting ? "Signing in..." : "Sign In"}
                  </Button>
                </form>
              </TabsContent>

              <TabsContent value="signup">
                <form onSubmit={handleSignup} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="name">Full Name</Label>
                    <Input
                      id="name"
                      type="text"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="Enter your full name"
                      required
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="signup-email">Email</Label>
                    <Input
                      id="signup-email"
                      type="email"
                      value={signupEmail}
                      onChange={(e) => setSignupEmail(e.target.value)}
                      placeholder="Enter your email"
                      required
                    />
                  </div>

                  <div className="space-y-3">
                    <Label>Select Your Role</Label>
                    <RadioGroup value={selectedRole} onValueChange={(value: any) => setSelectedRole(value)}>
                      {(Object.keys(roleIcons) as Array<keyof typeof roleIcons>).map((role) => {
                        const Icon = roleIcons[role];
                        return (
                          <div key={role} className="flex items-center space-x-3 p-3 border rounded-lg hover:bg-medical-light/30 cursor-pointer">
                            <RadioGroupItem value={role} id={role} />
                            <Icon className="w-5 h-5 text-primary" />
                            <div className="flex-1">
                              <Label htmlFor={role} className="font-medium capitalize cursor-pointer">
                                {role}
                              </Label>
                              <p className="text-xs text-muted-foreground">
                                {roleDescriptions[role]}
                              </p>
                            </div>
                          </div>
                        );
                      })}
                    </RadioGroup>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="signup-password">Password</Label>
                    <Input
                      id="signup-password"
                      type="password"
                      value={signupPassword}
                      onChange={(e) => setSignupPassword(e.target.value)}
                      placeholder="Create a password"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="confirm-password">Confirm Password</Label>
                    <Input
                      id="confirm-password"
                      type="password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      placeholder="Confirm your password"
                      required
                    />
                  </div>
                  <Button 
                    type="submit" 
                    className="w-full bg-gradient-to-r from-accent to-accent-hover hover:from-accent-hover hover:to-accent"
                    disabled={isSubmitting || loading || signupPassword !== confirmPassword}
                  >
                    {isSubmitting ? "Creating Account..." : "Create Account"}
                  </Button>
                </form>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default AuthPage;