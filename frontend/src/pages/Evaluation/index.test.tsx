import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';

// ── Mocks ────────────────────────────────────────────────────────────
vi.mock('../../api/evaluation', () => ({
  getEvaluation: vi.fn(),
  getEvaluationLockStatus: vi.fn(),
}));

vi.mock('../../api/consultation', () => ({
  getConsultationDetail: vi.fn(),
}));

// Mock the hook with controllable state via a callback pattern
let hookStateOverride: Record<string, unknown> = {};
let hookStartFn = vi.fn();
let hookCancelFn = vi.fn();
let hookResumeFn = vi.fn();

vi.mock('../../hooks/useEvaluationJob', () => ({
  useEvaluationJob: () => ({
    state: {
      runId: null,
      status: 'idle',
      progress: 0,
      message: '',
      errorCode: null,
      cancelRequested: false,
      isActive: false,
      ...hookStateOverride,
    },
    start: hookStartFn,
    cancel: hookCancelFn,
    resume: hookResumeFn,
  }),
}));

vi.mock('../../components', () => ({
  ScoreDisplay: ({ score }: { score: number }) => <span>{score}</span>,
  DimensionRadar: () => <div data-testid="radar" />,
  getScoreColor: () => '#1677ff',
  getScoreLevel: (s: number) => ({ text: s > 80 ? '优秀' : '良好', color: '#1677ff' }),
}));

import { getEvaluation, getEvaluationLockStatus } from '../../api/evaluation';
import { getConsultationDetail } from '../../api/consultation';
import EvaluationPage from './index';

const mockedGetEval = getEvaluation as ReturnType<typeof vi.fn>;
const mockedGetLock = getEvaluationLockStatus as ReturnType<typeof vi.fn>;
const mockedGetConsult = getConsultationDetail as ReturnType<typeof vi.fn>;

const renderPage = (consultationId = '1') =>
  render(
    <MemoryRouter initialEntries={[`/evaluation/${consultationId}`]}>
      <Routes>
        <Route path="/evaluation/:id" element={<EvaluationPage />} />
      </Routes>
    </MemoryRouter>,
  );

describe('EvaluationPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    hookStateOverride = {};
    hookStartFn = vi.fn();
    hookCancelFn = vi.fn();
    hookResumeFn = vi.fn();
    // Default: no existing evaluation
    mockedGetEval.mockRejectedValue(new Error('not found'));
    mockedGetLock.mockResolvedValue({ is_active: false, run_id: null });
    mockedGetConsult.mockResolvedValue({
      id: 1,
      diagnosis: '感冒',
      treatment_plan: '多休息',
      messages: [],
    });
  });

  it('点击"生成评估"后调用 hook start', async () => {
    renderPage();

    await waitFor(() => {
      expect(screen.getByText('暂无评估报告')).toBeTruthy();
    });

    const btn = screen.getByText('生成评估报告');
    fireEvent.click(btn);

    expect(hookStartFn).toHaveBeenCalled();
  });

  it('终态后加载报告', async () => {
    mockedGetEval.mockResolvedValue({
      id: 1,
      consultation_id: 1,
      inquiry_score: 85,
      inquiry_analysis: '良好',
      knowledge_score: 78,
      knowledge_analysis: '尚可',
      humanistic_score: 90,
      humanistic_analysis: '优秀',
      diagnosis_score: 80,
      diagnosis_analysis: '准确',
      treatment_score: 82,
      treatment_analysis: '合理',
      total_score: 83,
      overall_summary: '综合表现良好',
      improvement_suggestions: '一、加强问诊\n问题描述：问诊不够全面\n改进方法：增加既往史询问',
      evaluation_status: 'completed',
      retrieval_status: 'sufficient',
      evidence_stance: 'supports',
      human_review_needed: false,
      citation_data: null,
      created_at: '2026-01-01',
    });

    renderPage();

    await waitFor(() => {
      expect(screen.getByText('评估报告')).toBeTruthy();
    });
  });

  it('刷新页面可恢复 active run（hook 检测到 active lock 时调用 resume）', async () => {
    mockedGetEval.mockRejectedValue(new Error('not found'));
    mockedGetLock.mockResolvedValue({
      is_active: true,
      run_id: 'active-run-123',
      consultation_id: 1,
    });

    renderPage();

    await waitFor(() => {
      // The page should detect the active lock and call resume()
      expect(hookResumeFn).toHaveBeenCalled();
    });
  });
});
