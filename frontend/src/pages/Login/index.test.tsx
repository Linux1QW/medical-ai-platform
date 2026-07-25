import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { login } from '../../api/auth';
import LoginPage from './index';

vi.mock('../../api/auth', () => ({
  login: vi.fn(),
}));

const mockedLogin = vi.mocked(login);

const renderLogin = () =>
  render(
    <MemoryRouter initialEntries={['/login']}>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/dashboard" element={<div>dashboard-page</div>} />
      </Routes>
    </MemoryRouter>,
  );

describe('登录页', () => {
  beforeEach(() => {
    sessionStorage.clear();
    vi.clearAllMocks();
  });

  it('渲染标题与表单控件', () => {
    renderLogin();
    expect(screen.getByText('临床问诊评估平台')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('用户名')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('密码')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /登\s*录/ })).toBeInTheDocument();
  });

  it('空表单提交触发必填校验，不调用登录接口', async () => {
    const user = userEvent.setup();
    renderLogin();
    await user.click(screen.getByRole('button', { name: /登\s*录/ }));

    expect(await screen.findByText('请输入用户名')).toBeInTheDocument();
    expect(await screen.findByText('请输入密码')).toBeInTheDocument();
    expect(mockedLogin).not.toHaveBeenCalled();
  });

  it('登录成功后保存凭据并跳转 dashboard', async () => {
    mockedLogin.mockResolvedValue({
      access_token: 'jwt-token',
      token_type: 'bearer',
      user: { id: 1, username: 'doctor01', role: 'doctor' },
    } as never);

    const user = userEvent.setup();
    renderLogin();
    await user.type(screen.getByPlaceholderText('用户名'), 'doctor01');
    await user.type(screen.getByPlaceholderText('密码'), 'secret123');
    await user.click(screen.getByRole('button', { name: /登\s*录/ }));

    expect(await screen.findByText('dashboard-page')).toBeInTheDocument();
    expect(mockedLogin).toHaveBeenCalledWith({ username: 'doctor01', password: 'secret123' });
    expect(sessionStorage.getItem('token')).toBe('jwt-token');
  });

  it('登录失败时在密码项展示后端错误信息', async () => {
    mockedLogin.mockRejectedValue({
      response: { data: { message: '用户名或密码错误' } },
    });

    const user = userEvent.setup();
    renderLogin();
    await user.type(screen.getByPlaceholderText('用户名'), 'doctor01');
    await user.type(screen.getByPlaceholderText('密码'), 'wrong-pass');
    await user.click(screen.getByRole('button', { name: /登\s*录/ }));

    expect(await screen.findByText('用户名或密码错误')).toBeInTheDocument();
    await waitFor(() => expect(sessionStorage.getItem('token')).toBeNull());
  });
});
