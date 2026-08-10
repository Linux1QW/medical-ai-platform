/**
 * V1.1 Production Closure E2E 测试
 *
 * 完整评估 → 复核闭环流程（串行单测）：
 * 1. doctor 登录并打开已完成的问诊
 * 2. 提交评估并期望 202
 * 3. 等待状态变为 needs_review
 * 4. 以管理员身份复核
 * 5. 等待状态变为 reviewed
 */

import { test, expect, Page } from '@playwright/test';

// ──────────────────────────────────────────
// 测试配置
// ──────────────────────────────────────────
const E2E_DOCTOR_USER = process.env.E2E_DOCTOR_USER || 'doctor_v11';
const E2E_ADMIN_USER = process.env.E2E_ADMIN_USER || 'admin_v11';
const E2E_PASSWORD = process.env.E2E_PASSWORD || 'e2e_test_password_2026';

// ──────────────────────────────────────────
// 辅助函数
// ──────────────────────────────────────────
async function login(page: Page, username: string, password: string) {
  await page.goto('/login');
  await page.getByTestId('username-input').fill(username);
  await page.getByTestId('password-input').fill(password);
  await page.getByRole('button', { name: '登录' }).click();
  await expect(page).toHaveURL(/\/dashboard(?:\/|$)/, { timeout: 10_000 });
}

async function loginAndOpenCompletedConsultation(page: Page): Promise<string> {
  await login(page, E2E_DOCTOR_USER, E2E_PASSWORD);

  // 导航到问诊列表
  await page.goto('/consultations');
  await page.waitForLoadState('networkidle');

  // 找到 E2E 测试问诊（已结束状态）
  const consultationLink = page.locator('a[href*="/consultations/"]').first();
  const href = await consultationLink.getAttribute('href');
  const consultationId = href?.match(/\/consultations\/(\d+)/)?.[1] || '1';

  // 进入问诊详情
  await page.goto(`/evaluation/${consultationId}`);
  await page.waitForLoadState('networkidle');

  // 断言页面加载成功
  await expect(page.getByTestId('evaluation-page')).toBeVisible({ timeout: 5_000 });

  return consultationId;
}

async function submitEvaluationAndExpect202(page: Page): Promise<string> {
  // 监听 POST 请求
  const postPromise = page.waitForResponse(
    res => res.url().includes('/api/v1/evaluations') && res.request().method() === 'POST',
    { timeout: 10_000 }
  );

  // 点击生成评估按钮
  await page.getByTestId('generate-evaluation-button').click();

  // 断言 POST 返回 202
  const response = await postPromise;
  expect(response.status()).toBe(202);

  // 从响应中提取 runId
  const body = await response.json();
  const runId = body.run_id || body.id || 'unknown';

  // 断言 UI 显示运行状态
  await expect(page.getByTestId('evaluation-status')).toBeVisible({ timeout: 15_000 });

  return runId;
}

async function expectRunStatus(page: Page, runId: string, expectedStatus: string) {
  // 轮询等待状态变为期望值
  await expect(page.getByTestId('evaluation-status')).toContainText(expectedStatus, { timeout: 30_000 });
}

async function reviewAsAdmin(
  page: Page,
  runId: string,
  review: { knowledgeScore: number; feedback: string }
) {
  await login(page, E2E_ADMIN_USER, E2E_PASSWORD);

  // 导航到复核工作台
  await page.goto('/admin/reviews');
  await page.waitForLoadState('networkidle');

  // 断言复核页面加载成功
  await expect(page.getByTestId('review-page')).toBeVisible({ timeout: 5_000 });

  // 打开对应报告
  const reportLink = page.locator('a[href*="/reviews/"], tr:has-text("doctor_v11")').first();
  await reportLink.click();
  await page.waitForLoadState('networkidle');

  // 填写反馈
  await page.getByTestId('feedback-input').fill(review.feedback);

  // 调整 knowledge score
  await page.getByTestId('knowledge-score-input').fill(String(review.knowledgeScore));

  // 提交
  await page.getByRole('button', { name: '提交' }).click();

  // 等待提交成功
  await expect(page).toHaveURL(/\/admin\/reviews/, { timeout: 10_000 });
}

// ──────────────────────────────────────────
// E2E 测试用例
// ──────────────────────────────────────────
test.describe('V1.1 Evaluation and Review Closure', () => {
  test('doctor submission reaches reviewed after admin review', async ({ page }) => {
    // 1. 登录并打开已完成的问诊
    const consultationId = await loginAndOpenCompletedConsultation(page);

    // 2. 提交评估并期望 202
    const runId = await submitEvaluationAndExpect202(page);

    // 3. 等待状态变为 needs_review
    await expectRunStatus(page, runId, 'needs_review');

    // 4. 以管理员身份复核
    await reviewAsAdmin(page, runId, { knowledgeScore: 75, feedback: '证据已人工核验' });

    // 5. 等待状态变为 reviewed
    await expectRunStatus(page, runId, 'reviewed');
  });
});

// ──────────────────────────────────────────
// Fault Case 测试（非浏览器，由 run_v11_fault_matrix.py 执行）
// ──────────────────────────────────────────
test.describe('V1.1 Fault Cases', () => {
  test.skip('dispatcher outage recovery', async () => {
    // 此测试需要在 docker compose 环境中运行
    // 停止 evaluation-dispatcher，提交评估，恢复 dispatcher，验证完成
  });

  test.skip('redis-state fail-closed', async () => {
    // 停止 redis-state，验证认证/API 返回 503
  });

  test.skip('cache saturation', async () => {
    // 写满 redis-cache，验证核心功能不受影响
  });

  test.skip('execution-owner fencing', async () => {
    // Worker A 停止心跳，Worker B 接管，验证只有一份 Evaluation
  });
});
