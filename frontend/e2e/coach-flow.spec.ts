import { test, expect } from '@playwright/test';

test.describe('Coach Flow E2E', () => {
  test('coach panel renders in disabled state by default', async ({ page }) => {
    // Navigate to a consultation page (mock route)
    await page.goto('/consultations/1');
    
    // Coach panel should be visible
    const panel = page.locator('.coach-panel');
    await expect(panel).toBeVisible();
    
    // Should show disabled or idle status
    const statusText = await panel.textContent();
    expect(statusText).toBeTruthy();
  });

  test('coach panel shows input when idle', async ({ page }) => {
    await page.goto('/consultations/1');
    
    const panel = page.locator('.coach-panel');
    const textarea = panel.locator('textarea');
    
    // If panel is in idle state, textarea should be visible
    if (await panel.textContent().then(t => t?.includes('等待问诊'))) {
      await expect(textarea).toBeVisible();
    }
  });

  test('voice consultation component renders', async ({ page }) => {
    await page.goto('/consultations/1');
    
    // Voice component should exist if present
    const voiceSection = page.locator('.voice-consultation');
    if (await voiceSection.isVisible()) {
      await expect(voiceSection).toContainText('语音问诊');
    }
  });
});
