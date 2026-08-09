import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright 配置 for V1.1 Production Closure E2E
 * 
 * 测试完整评估 → 复核闭环流程
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: false, // E2E 测试需要顺序执行
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1, // 单 worker 确保顺序
  reporter: [
    ['html', { open: 'never' }],
    ['list'],
  ],
  
  /* 共享测试配置 */
  use: {
    baseURL: process.env.E2E_BASE_URL || 'http://localhost:8080',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },

  /* 项目配置 */
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  /* 开发服务器（不需要，E2E 使用 docker compose） */
  // webServer: {
  //   command: 'npm run dev',
  //   url: 'http://localhost:5173',
  //   reuseExistingServer: !process.env.CI,
  // },

  /* 超时配置 */
  timeout: 60_000, // 单个测试超时
  expect: {
    timeout: 10_000, // 断言超时
  },
});
