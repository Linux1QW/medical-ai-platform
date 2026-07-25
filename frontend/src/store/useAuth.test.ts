import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';

import type { User } from '../types';
import { useAuth } from './useAuth';

const fakeUser = { id: 1, username: 'doctor01', role: 'doctor' } as unknown as User;

describe('useAuth', () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  it('初始状态未登录', () => {
    const { result } = renderHook(() => useAuth());
    expect(result.current.isLoggedIn).toBe(false);
    expect(result.current.user).toBeNull();
    expect(result.current.token).toBeNull();
  });

  it('saveAuth 后写入 sessionStorage 并变为已登录', () => {
    const { result } = renderHook(() => useAuth());
    act(() => result.current.saveAuth('jwt-token', fakeUser));

    expect(result.current.isLoggedIn).toBe(true);
    expect(result.current.isAdmin).toBe(false);
    expect(sessionStorage.getItem('token')).toBe('jwt-token');
    expect(JSON.parse(sessionStorage.getItem('user')!)).toMatchObject({ username: 'doctor01' });
  });

  it('logout 清空状态与 sessionStorage', () => {
    const { result } = renderHook(() => useAuth());
    act(() => result.current.saveAuth('jwt-token', fakeUser));
    act(() => result.current.logout());

    expect(result.current.isLoggedIn).toBe(false);
    expect(sessionStorage.getItem('token')).toBeNull();
    expect(sessionStorage.getItem('user')).toBeNull();
  });

  it('从 sessionStorage 恢复登录态；损坏数据时安全回退', () => {
    sessionStorage.setItem('token', 'jwt-token');
    sessionStorage.setItem('user', JSON.stringify({ ...fakeUser, role: 'admin' }));
    const { result } = renderHook(() => useAuth());
    expect(result.current.isLoggedIn).toBe(true);
    expect(result.current.isAdmin).toBe(true);

    sessionStorage.setItem('user', '{broken json');
    const { result: broken } = renderHook(() => useAuth());
    expect(broken.current.user).toBeNull();
    expect(sessionStorage.getItem('user')).toBeNull();
  });
});
