/* vitest 全局初始化：jest-dom 断言 + jsdom 缺失的浏览器 API 补齐 */
import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';

// 未开启 vitest globals，RTL 不会自动注册 cleanup，需显式卸载避免多次 render 累积
afterEach(() => {
  cleanup();
});

// antd 响应式组件依赖 matchMedia，jsdom 未实现
if (!window.matchMedia) {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
}

// antd（rc-resize-observer）依赖 ResizeObserver
if (!window.ResizeObserver) {
  class ResizeObserverMock {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  window.ResizeObserver = ResizeObserverMock as unknown as typeof ResizeObserver;
}
