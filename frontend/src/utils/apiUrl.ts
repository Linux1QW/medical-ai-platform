const configuredBase = import.meta.env.VITE_API_BASE_URL?.trim();
export const API_BASE_URL = (configuredBase || '/api/v1').replace(/\/$/, '');

export function toWebSocketUrl(path: string): string {
  const httpUrl = /^https?:\/\//.test(path)
    ? new URL(path)
    : path.startsWith('/')
      ? new URL(path, window.location.origin)
      : new URL(`${API_BASE_URL}/${path.replace(/^\//, '')}`, window.location.origin);
  httpUrl.protocol = httpUrl.protocol === 'https:' ? 'wss:' : 'ws:';
  return httpUrl.toString();
}
