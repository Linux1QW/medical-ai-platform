import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright 配置 for V1.2 Coach Intelligence E2E
 *
 * 测试 Coach 建议 → 反馈 → 持久化闭环流程
 * 双 Backend 实例 + 共享 MySQL/Redis 基础设施
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

  /* 环境变量传递给测试 */
  metadata: {
    backendA: process.env.E2E_BACKEND_A || 'http://localhost:8000',
    backendB: process.env.E2E_BACKEND_B || 'http://localhost:8001',
    consultationId: process.env.E2E_CONSULTATION_ID || '1',
    doctorUser: process.env.E2E_DOCTOR_USER || 'doctor_v12',
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
  timeout: 120_000, // 单个测试超时 (V1.2 Coach SSE + 双 Backend 验证)
  expect: {
    timeout: 15_000, // 断言超时
  },
});
