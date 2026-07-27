import { message } from 'antd';
import type { InternalAxiosRequestConfig } from 'axios';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('antd', () => ({
  message: {
    error: vi.fn(),
    warning: vi.fn(),
  },
}));

type RequestFulfilled = (
  config: InternalAxiosRequestConfig,
) => InternalAxiosRequestConfig;

interface ResponseHandlers {
  fulfilled: (response: { data: unknown }) => unknown;
  rejected: (error: unknown) => Promise<unknown>;
}

/** 每次重新加载模块，重置 isRedirectingTo401 等模块级状态 */
const loadInterceptors = async () => {
  vi.resetModules();
  const request = (await import('./request')).default;
  const requestHandlers = (
    request.interceptors.request as unknown as {
      handlers: { fulfilled: RequestFulfilled }[];
    }
  ).handlers;
  const responseHandlers = (
    request.interceptors.response as unknown as { handlers: ResponseHandlers[] }
  ).handlers;
  return {
    onRequest: requestHandlers[0].fulfilled,
    onResponse: responseHandlers[0],
  };
};

const makeConfig = (url: string): InternalAxiosRequestConfig =>
  ({ url, headers: {} }) as unknown as InternalAxiosRequestConfig;

describe('request 拦截器', () => {
  beforeEach(() => {
    sessionStorage.clear();
    vi.clearAllMocks();
  });

  it('存在 token 时注入 Authorization 头', async () => {
    sessionStorage.setItem('token', 'jwt-abc');
    const { onRequest } = await loadInterceptors();
    const config = onRequest(makeConfig('/patients'));
    expect(config.headers.Authorization).toBe('Bearer jwt-abc');
  });

  it('无 token 时不注入 Authorization 头', async () => {
    const { onRequest } = await loadInterceptors();
    const config = onRequest(makeConfig('/patients'));
    expect(config.headers.Authorization).toBeUndefined();
  });

  it('评估类接口使用 300s 超时，普通接口 60s', async () => {
    const { onRequest } = await loadInterceptors();
    expect(onRequest(makeConfig('/evaluation/1')).timeout).toBe(300000);
    expect(onRequest(makeConfig('/reports/2')).timeout).toBe(300000);
    expect(onRequest(makeConfig('/patients')).timeout).toBe(60000);
  });

  it('响应成功时直接返回 response.data', async () => {
    const { onResponse } = await loadInterceptors();
    expect(onResponse.fulfilled({ data: { items: [1, 2] } })).toEqual({
      items: [1, 2],
    });
  });

  it('超时错误提示耐心等待', async () => {
    const { onResponse } = await loadInterceptors();
    await expect(
      onResponse.rejected({
        code: 'ECONNABORTED',
        message: 'timeout of 60000ms exceeded',
        config: { url: '/patients' },
      }),
    ).rejects.toBeTruthy();
    expect(message.warning).toHaveBeenCalledWith('后端仍在处理，请勿刷新，请耐心等待');
    expect(message.error).not.toHaveBeenCalled();
  });

  it('普通错误优先展示后端 message/detail', async () => {
    const { onResponse } = await loadInterceptors();
    await expect(
      onResponse.rejected({
        response: { status: 500, data: { detail: '服务器内部错误' } },
        config: { url: '/patients' },
      }),
    ).rejects.toBeTruthy();
    expect(message.error).toHaveBeenCalledWith('服务器内部错误');
  });

  it('无后端信息时展示默认"请求失败"', async () => {
    const { onResponse } = await loadInterceptors();
    await expect(
      onResponse.rejected({
        response: { status: 500, data: {} },
        config: { url: '/patients' },
      }),
    ).rejects.toBeTruthy();
    expect(message.error).toHaveBeenCalledWith('请求失败');
  });

  it('401 时清除会话凭据', async () => {
    sessionStorage.setItem('token', 'jwt-abc');
    sessionStorage.setItem('user', '{"id":1}');
    const { onResponse } = await loadInterceptors();
    await expect(
      onResponse.rejected({
        response: { status: 401, data: {} },
        config: { url: '/patients' },
      }),
    ).rejects.toBeTruthy();
    expect(sessionStorage.getItem('token')).toBeNull();
    expect(sessionStorage.getItem('user')).toBeNull();
  });

  it('登录接口的 401 不触发凭据清除（由登录页自行展示错误）', async () => {
    sessionStorage.setItem('token', 'stale-token');
    const { onResponse } = await loadInterceptors();
    await expect(
      onResponse.rejected({
        response: { status: 401, data: { message: '用户名或密码错误' } },
        config: { url: '/auth/login' },
      }),
    ).rejects.toBeTruthy();
    expect(sessionStorage.getItem('token')).toBe('stale-token');
  });
});
