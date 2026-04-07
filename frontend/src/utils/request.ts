import axios from 'axios';
import { message } from 'antd';

const request = axios.create({
  baseURL: '/api/v1',
  timeout: 300000,
});

// 评估类接口路径前缀或关键字
const EVALUATION_API_KEYWORDS = ['/evaluation', '/evaluate', '/reports'];

request.interceptors.request.use((config) => {
  const token = sessionStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }

  // 分级超时策略：评估类接口300000ms，普通查询类接口60000ms
  const isEvaluationApi = EVALUATION_API_KEYWORDS.some(keyword => config.url?.includes(keyword));
  config.timeout = isEvaluationApi ? 300000 : 60000;

  return config;
});

let isRedirectingTo401 = false;

request.interceptors.response.use(
  (response) => response.data,
  (error) => {
    if (error.code === 'ECONNABORTED' && error.message.includes('timeout')) {
      message.warning('后端仍在处理，请勿刷新，请耐心等待');
    } else {
      const data = error.response?.data;
      const msg = data?.message || data?.detail || '请求失败';
      message.error(msg);
    }

    const isLoginRequest = error.config?.url?.includes('/auth/login');
    if (error.response?.status === 401 && !isRedirectingTo401 && !isLoginRequest) {
      isRedirectingTo401 = true;
      sessionStorage.removeItem('token');
      sessionStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  },
);

export default request;
