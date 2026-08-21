// App shell + the authentication guard.
//
// While the session is being restored we show a placeholder; then we either
// render the sign-in screen or the authenticated app. With a single protected
// area, the guard is a simple conditional render; URL-level routing arrives when
// there are multiple protected pages.

import { AuthProvider, useAuth } from "./auth/AuthContext";
import { AuthPage } from "./components/AuthPage";
import { Dashboard } from "./components/Dashboard";

function Gate(): React.JSX.Element {
  const { user, loading } = useAuth();
  if (loading) return <p>Loading…</p>;
  return user ? <Dashboard /> : <AuthPage />;
}

export function App(): React.JSX.Element {
  return (
    <AuthProvider>
      <Gate />
    </AuthProvider>
  );
}
