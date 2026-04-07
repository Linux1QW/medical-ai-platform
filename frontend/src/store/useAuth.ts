import { useCallback, useState } from 'react';
import type { User } from '../types';

export function useAuth() {
  const [user, setUser] = useState<User | null>(() => {
    try {
      const stored = sessionStorage.getItem('user');
      return stored ? JSON.parse(stored) : null;
    } catch {
      sessionStorage.removeItem('user');
      return null;
    }
  });

  const [token, setToken] = useState<string | null>(() => sessionStorage.getItem('token'));

  const saveAuth = useCallback((accessToken: string, userData: User) => {
    sessionStorage.setItem('token', accessToken);
    sessionStorage.setItem('user', JSON.stringify(userData));
    setToken(accessToken);
    setUser(userData);
  }, []);

  const logout = useCallback(() => {
    sessionStorage.removeItem('token');
    sessionStorage.removeItem('user');
    setToken(null);
    setUser(null);
  }, []);

  const isLoggedIn = !!token && !!user;
  const isAdmin = user?.role === 'admin';

  return { user, token, isLoggedIn, isAdmin, saveAuth, logout };
}
