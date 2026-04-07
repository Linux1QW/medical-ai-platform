import request from '../utils/request';
import type { Token, User } from '../types';

export const login = (data: { username: string; password: string }): Promise<Token> =>
  request.post('/auth/login', data);

export const register = (data: {
  username: string;
  password: string;
  real_name?: string;
  department?: string;
}): Promise<User> => request.post('/auth/register', data);

export const getMe = (): Promise<User> => request.get('/auth/me');

export const updateProfile = (data: {
  real_name?: string;
  department?: string;
  email?: string;
}): Promise<User> => request.put('/auth/profile', data);
