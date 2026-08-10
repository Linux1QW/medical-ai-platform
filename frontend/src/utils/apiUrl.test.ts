import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

describe('apiUrl', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('默认 API_BASE_URL 为 /api/v1', async () => {
    const { API_BASE_URL } = await import('./apiUrl');
    expect(API_BASE_URL).toBe('/api/v1');
  });

  it('读取 VITE_API_BASE_URL 环境变量并去除末尾斜杠', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.com/v2/');
    const { API_BASE_URL } = await import('./apiUrl');
    expect(API_BASE_URL).toBe('https://api.example.com/v2');
  });

  it('空白环境变量回退到默认值', async () => {
    vi.stubEnv('VITE_API_BASE_URL', '   ');
    const { API_BASE_URL } = await import('./apiUrl');
    expect(API_BASE_URL).toBe('/api/v1');
  });

  describe('toWebSocketUrl', () => {
    it('相对路径基于 origin 转换', async () => {
      const { toWebSocketUrl } = await import('./apiUrl');
      const result = toWebSocketUrl('/api/v1/evaluations/ws/runs/abc');
      expect(result).toMatch(/^ws:\/\//);
      expect(result).toContain('/api/v1/evaluations/ws/runs/abc');
    });

    it('https 绝对路径转换为 wss', async () => {
      const { toWebSocketUrl } = await import('./apiUrl');
      const result = toWebSocketUrl('https://api.example.com/api/v1/evaluations/ws/runs/abc');
      expect(result).toBe('wss://api.example.com/api/v1/evaluations/ws/runs/abc');
    });

    it('http 绝对路径转换为 ws', async () => {
      const { toWebSocketUrl } = await import('./apiUrl');
      const result = toWebSocketUrl('http://localhost:8000/api/v1/evaluations/ws/runs/abc');
      expect(result).toBe('ws://localhost:8000/api/v1/evaluations/ws/runs/abc');
    });

    it('非绝对路径拼接 API_BASE_URL', async () => {
      vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.com/v2');
      const { toWebSocketUrl } = await import('./apiUrl');
      const result = toWebSocketUrl('evaluations/ws/runs/abc');
      expect(result).toBe('wss://api.example.com/v2/evaluations/ws/runs/abc');
    });
  });
});
