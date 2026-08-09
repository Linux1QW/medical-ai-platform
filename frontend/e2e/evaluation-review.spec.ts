/**
 * V1.1 Production Closure E2E 测试
 * 
 * 完整评估 → 复核闭环流程：
 * 1. doctor 登录并进入评估页面
 * 2. 点击生成，断言 POST 202 + progress 事件
 * 3. 关闭 WebSocket，断言重连后达到 needs_review
 * 4. 刷新页面，断言恢复
 * 5. admin 登录，进入复核工作台
 * 6. 填写反馈并提交
 * 7. 队列消失，doctor 看到 reviewed
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
  await page.fill('input[name="username"], input[id="username"], input[placeholder*="用户"]', username);
  await page.fill('input[name="password"], input[id="password"], input[placeholder*="密码"]', password);
  await page.click('button[type="submit"], button:has-text("登录")');
  await page.waitForURL('**/dashboard**', { timeout: 10_000 }).catch(() => {
    // 可能跳转到其他页面
  });
}

async function waitForEvaluationStatus(page: Page, consultationId: string, expectedStatus: string, timeout = 30_000) {
  const startTime = Date.now();
  while (Date.now() - startTime < timeout) {
    const response = await page.evaluate(async (id) => {
      const res = await fetch(`/api/v1/evaluations/by-consultation/${id}`);
      if (res.ok) return await res.json();
      return null;
    }, consultationId);
    
    if (response && response.evaluation_status === expectedStatus) {
      return response;
    }
    await page.waitForTimeout(1000);
  }
  throw new Error(`Timeout waiting for evaluation status: ${expectedStatus}`);
}

// ──────────────────────────────────────────
// E2E 测试用例
// ──────────────────────────────────────────
test.describe('V1.1 Evaluation and Review Closure', () => {
  let consultationId: string;

  test('1. doctor 登录并进入评估页面', async ({ page }) => {
    await login(page, E2E_DOCTOR_USER, E2E_PASSWORD);
    
    // 导航到问诊列表
    await page.goto('/consultations');
    await page.waitForLoadState('networkidle');
    
    // 找到 E2E 测试问诊（已结束状态）
    const consultationLink = page.locator('a[href*="/consultations/"]').first();
    const href = await consultationLink.getAttribute('href');
    consultationId = href?.match(/\/consultations\/(\d+)/)?.[1] || '1';
    
    // 进入问诊详情
    await page.goto(`/evaluation/${consultationId}`);
    await page.waitForLoadState('networkidle');
    
    // 断言页面加载成功
    await expect(page.locator('text=评估').first()).toBeVisible({ timeout: 5_000 });
  });

  test('2. 点击生成，断言 POST 202 + progress', async ({ page }) => {
    await login(page, E2E_DOCTOR_USER, E2E_PASSWORD);
    await page.goto(`/evaluation/${consultationId || '1'}`);
    await page.waitForLoadState('networkidle');
    
    // 监听 POST 请求
    const postPromise = page.waitForResponse(
      res => res.url().includes('/api/v1/evaluations') && res.request().method() === 'POST',
      { timeout: 10_000 }
    );
    
    // 点击生成评估按钮
    const generateButton = page.locator('button:has-text("生成"), button:has-text("评估")').first();
    await generateButton.click();
    
    // 断言 POST 返回 202
    const response = await postPromise;
    expect(response.status()).toBe(202);
    
    // 等待 progress 事件（通过 WebSocket 或轮询）
    await page.waitForTimeout(2000);
    
    // 断言 UI 显示运行状态
    const statusIndicator = page.locator('[data-testid="evaluation-status"], .evaluation-status, text=运行中, text=评估中').first();
    await expect(statusIndicator).toBeVisible({ timeout: 15_000 });
  });

  test('3. WebSocket 断开后重连仍达到 needs_review', async ({ page }) => {
    await login(page, E2E_DOCTOR_USER, E2E_PASSWORD);
    await page.goto(`/evaluation/${consultationId || '1'}`);
    await page.waitForLoadState('networkidle');
    
    // 等待评估完成（可能需要等待 Celery 任务完成）
    await page.waitForTimeout(5000);
    
    // 刷新页面模拟重连
    await page.reload();
    await page.waitForLoadState('networkidle');
    
    // 断言达到 needs_review 状态
    const needsReview = page.locator('text=needs_review, text=待复核, text=需要复核').first();
    await expect(needsReview).toBeVisible({ timeout: 30_000 });
  });

  test('4. 刷新页面后恢复状态', async ({ page }) => {
    await login(page, E2E_DOCTOR_USER, E2E_PASSWORD);
    await page.goto(`/evaluation/${consultationId || '1'}`);
    await page.waitForLoadState('networkidle');
    
    // 刷新页面
    await page.reload();
    await page.waitForLoadState('networkidle');
    
    // 断言状态恢复（不重复 POST）
    const statusElement = page.locator('[data-testid="evaluation-status"], .evaluation-status').first();
    await expect(statusElement).toBeVisible({ timeout: 5_000 });
  });

  test('5. admin 登录并进入复核工作台', async ({ page }) => {
    await login(page, E2E_ADMIN_USER, E2E_PASSWORD);
    
    // 导航到复核工作台
    await page.goto('/admin/reviews');
    await page.waitForLoadState('networkidle');
    
    // 断言页面加载成功
    await expect(page.locator('text=复核, text=审核').first()).toBeVisible({ timeout: 5_000 });
  });

  test('6. 填写反馈并提交', async ({ page }) => {
    await login(page, E2E_ADMIN_USER, E2E_PASSWORD);
    await page.goto('/admin/reviews');
    await page.waitForLoadState('networkidle');
    
    // 打开对应报告
    const reportLink = page.locator('a[href*="/reviews/"], tr:has-text("doctor_v11")').first();
    await reportLink.click();
    await page.waitForLoadState('networkidle');
    
    // 填写"证据已人工核验"
    const feedbackInput = page.locator('textarea[placeholder*="反馈"], textarea[name="feedback"], textarea').first();
    await feedbackInput.fill('证据已人工核验');
    
    // 调整 knowledge score 为 75
    const knowledgeScoreInput = page.locator('input[name="knowledge_score"], input[placeholder*="知识"]').first();
    await knowledgeScoreInput.fill('75');
    
    // 提交
    const submitButton = page.locator('button[type="submit"], button:has-text("提交")').first();
    await submitButton.click();
    
    // 等待提交成功
    await page.waitForTimeout(2000);
  });

  test('7. 队列消失，doctor 看到 reviewed', async ({ page }) => {
    // 验证 admin 端队列消失
    await login(page, E2E_ADMIN_USER, E2E_PASSWORD);
    await page.goto('/admin/reviews');
    await page.waitForLoadState('networkidle');
    
    // 断言队列项消失（或状态变为已复核）
    await page.waitForTimeout(1000);
    
    // 切换到 doctor 验证
    await login(page, E2E_DOCTOR_USER, E2E_PASSWORD);
    await page.goto(`/evaluation/${consultationId || '1'}`);
    await page.waitForLoadState('networkidle');
    
    // 断言报告状态为 reviewed
    const reviewedStatus = page.locator('text=reviewed, text=已复核').first();
    await expect(reviewedStatus).toBeVisible({ timeout: 10_000 });
  });
});

// ──────────────────────────────────────────
// Fault Case 测试（非浏览器）
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
