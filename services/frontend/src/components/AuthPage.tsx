// Sign-in / sign-up screen shown when no user is authenticated.
//
// One form toggles between login and register. Google sign-in is a plain link:
// it's a full-page navigation to the backend, which runs the OAuth dance and
// redirects back with the auth cookies set.

import { useState, type FormEvent } from "react";

import { ApiError, googleLoginUrl } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { Alert, Button, FieldLabel, Input } from "./ui";

type Mode = "login" | "register";

export function AuthPage(): React.JSX.Element {
  const { login, register } = useAuth();
  const [mode, setMode] = useState<Mode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      if (mode === "login") {
        await login(email, password);
      } else {
        await register(email, password, displayName || undefined);
      }
    } catch (err) {
      setError(messageFor(err, mode));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-stone-50 px-4 py-12">
      <div className="w-full max-w-sm">
        <div className="mb-8 text-center">
          <h1 className="font-serif text-3xl font-semibold text-stone-900">Recap</h1>
          <p className="mt-1 text-sm text-stone-500">Your reading assistant.</p>
        </div>

        <div className="rounded-xl border border-stone-200 bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-stone-900">
            {mode === "login" ? "Log in" : "Create your account"}
          </h2>

          <form onSubmit={submit} aria-label={mode} className="mt-5 space-y-4">
            {mode === "register" && (
              <FieldLabel className="block space-y-1">
                <span>Name</span>
                <Input
                  type="text"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  autoComplete="name"
                />
              </FieldLabel>
            )}
            <FieldLabel className="block space-y-1">
              <span>Email</span>
              <Input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoComplete="email"
              />
            </FieldLabel>
            <FieldLabel className="block space-y-1">
              <span>Password</span>
              <Input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                minLength={mode === "register" ? 8 : undefined}
              />
            </FieldLabel>

            {error && <Alert>{error}</Alert>}

            <Button type="submit" variant="primary" disabled={submitting} className="w-full">
              {mode === "login" ? "Log in" : "Sign up"}
            </Button>
          </form>

          <div className="my-5 flex items-center gap-3 text-xs text-stone-400">
            <div className="h-px flex-1 bg-stone-200" />
            or
            <div className="h-px flex-1 bg-stone-200" />
          </div>

          <a
            href={googleLoginUrl()}
            className="flex w-full items-center justify-center rounded-lg border border-stone-300 px-4 py-2 text-sm font-medium text-stone-700 transition-colors hover:bg-stone-50"
          >
            Continue with Google
          </a>
        </div>

        <p className="mt-5 text-center text-sm text-stone-500">
          {mode === "login" ? "New here?" : "Already have an account?"}{" "}
          <button
            type="button"
            className="font-medium text-indigo-600 hover:text-indigo-700"
            onClick={() => {
              setMode(mode === "login" ? "register" : "login");
              setError(null);
            }}
          >
            {mode === "login" ? "Create an account" : "Log in"}
          </button>
        </p>
      </div>
    </main>
  );
}

/** Map an API error to a friendly message. */
function messageFor(err: unknown, mode: Mode): string {
  if (err instanceof ApiError) {
    if (err.code === "INVALID_CREDENTIALS") return "Incorrect email or password.";
    if (err.code === "USER_ALREADY_EXISTS") return "An account with this email already exists.";
    if (err.status === 422) return "Please enter a valid email and a password of at least 8 characters.";
    return err.message;
  }
  return mode === "login" ? "Could not log in. Please try again." : "Could not sign up. Please try again.";
}
