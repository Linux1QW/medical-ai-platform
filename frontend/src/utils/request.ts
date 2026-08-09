import axios from 'axios';
import { message } from 'antd';
import { API_BASE_URL } from './apiUrl';

const request = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

request.interceptors.request.use((config) => {
  const token = sessionStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
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
      // 带 error_code 的错误由调用方（hook/组件）自行处理，全局拦截器不重复弹
      const hasErrorCode = !!data?.error_code;
      if (!hasErrorCode) {
        const msg = data?.message || data?.detail || '请求失败';
        message.error(msg);
      }
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
