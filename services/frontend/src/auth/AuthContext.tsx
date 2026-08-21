// Authentication state for the SPA.
//
// The context holds the current user and the actions that change it. Auth itself
// lives in httpOnly cookies the browser sends automatically, so this never
// touches a token — it just tracks "who is signed in" by calling the API. On
// mount it probes /users/me to restore a session (e.g. after an OAuth redirect).

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import * as api from "../api/client";
import type { ProfileUpdate, User } from "../api/types";

interface AuthContextValue {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, displayName?: string) => Promise<void>;
  logout: () => Promise<void>;
  updateProfile: (patch: ProfileUpdate) => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }): React.JSX.Element {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // Restore any existing session on first load.
  useEffect(() => {
    let active = true;
    api
      .getMe()
      .then((me) => active && setUser(me))
      .catch(() => active && setUser(null))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, []);

  // apiFetch silently renews an expired access token from the refresh cookie;
  // when that renewal itself fails (refresh cookie missing/expired too), it
  // reports back here so the app drops to the sign-in screen instead of
  // leaving the user looking at a dead session with failing requests.
  useEffect(() => {
    api.setSessionExpiredHandler(() => setUser(null));
    return () => api.setSessionExpiredHandler(null);
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    setUser(await api.login(email, password));
  }, []);

  const register = useCallback(
    async (email: string, password: string, displayName?: string) => {
      // Register creates the account; log in to obtain the session cookies.
      await api.register(email, password, displayName);
      setUser(await api.login(email, password));
    },
    [],
  );

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } finally {
      setUser(null);
    }
  }, []);

  const updateProfile = useCallback(async (patch: ProfileUpdate) => {
    setUser(await api.updateProfile(patch));
  }, []);

  const value = useMemo(
    () => ({ user, loading, login, register, logout, updateProfile }),
    [user, loading, login, register, logout, updateProfile],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

/** Access the auth context; throws if used outside an `AuthProvider`. */
export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (ctx === undefined) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
