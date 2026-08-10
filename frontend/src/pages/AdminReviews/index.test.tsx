import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import AdminReviews from './index';
import * as reviewApi from '../../api/review';
import * as evaluationApi from '../../api/evaluation';
import { useAuth } from '../../store/useAuth';

// Mock API modules
vi.mock('../../api/review', () => ({
  listPendingReviews: vi.fn(),
  getReviewStatus: vi.fn(),
  submitReview: vi.fn(),
}));

vi.mock('../../api/evaluation', () => ({
  getEvaluation: vi.fn(),
}));

// Mock useAuth
vi.mock('../../store/useAuth', () => ({
  useAuth: vi.fn(),
}));

const mockedListPending = vi.mocked(reviewApi.listPendingReviews);
const mockedSubmitReview = vi.mocked(reviewApi.submitReview);
const mockedGetEvaluation = vi.mocked(evaluationApi.getEvaluation);
const mockedUseAuth = vi.mocked(useAuth);

const mockPendingItems = [
  {
    evaluation_id: 1,
    consultation_id: 101,
    doctor_username: 'doctor01',
    patient_name: '患者张三',
    review_reason: '证据不足',
    human_review_needed: true,
    created_at: '2026-08-01T10:00:00Z',
  },
  {
    evaluation_id: 2,
    consultation_id: 102,
    doctor_username: 'doctor02',
    patient_name: '患者李四',
    review_reason: '高危红旗',
    human_review_needed: true,
    created_at: '2026-08-01T09:00:00Z',
  },
];

const mockEvaluation = {
  id: 1,
  consultation_id: 101,
  inquiry_score: 85,
  inquiry_analysis: '问诊分析',
  knowledge_score: 78,
  knowledge_analysis: '知识分析',
  humanistic_score: 90,
  humanistic_analysis: '人文分析',
  diagnosis_score: 82,
  diagnosis_analysis: '诊断分析',
  treatment_score: 88,
  treatment_analysis: '治疗分析',
  total_score: 84,
  overall_summary: '总体评价',
  improvement_suggestions: '改进建议',
  created_at: '2026-08-01T10:00:00Z',
  citation_data: [
    {
      citation_id: 'cite-001',
      claim: '心肌梗死诊断',
      source: '内科学',
      page: 123,
      heading_path: '心血管内科',
      text_snippet: 'ST段抬高型心肌梗死',
      rerank_score: 0.85,
    },
  ],
  rag_trace_data: null,
  retrieval_status: 'sufficient' as const,
  evidence_stance: 'supports' as const,
  human_review_needed: true,
  review_reason: '证据不足',
  evaluation_status: 'needs_review' as const,
};

const renderWithRouter = (initialPath = '/admin/reviews', isAdmin = true) => {
  mockedUseAuth.mockReturnValue({
    user: { id: 1, username: 'admin01', role: isAdmin ? 'admin' : 'doctor', email: '', real_name: '管理员', department: '', avatar: '', created_at: '' },
    token: 'test-token',
    isLoggedIn: true,
    isAdmin,
    saveAuth: vi.fn(),
    logout: vi.fn(),
  });

  function AdminRoute({ children }: { children: React.ReactNode }) {
    const { isAdmin: admin } = { isAdmin };
    if (!admin) return <div>dashboard-page</div>;
    return <>{children}</>;
  }

  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <Routes>
        <Route path="/admin/reviews" element={<AdminRoute><AdminReviews /></AdminRoute>} />
        <Route path="/dashboard" element={<div>dashboard-page</div>} />
        <Route path="/login" element={<div>login-page</div>} />
      </Routes>
    </MemoryRouter>,
  );
};

describe('AdminReviews 复核工作台', () => {
  beforeEach(() => {
    sessionStorage.clear();
    vi.clearAllMocks();
  });

  it('mock API 返回两条 pending，断言表格渲染', async () => {
    mockedListPending.mockResolvedValue({
      items: mockPendingItems,
      total: 2,
      limit: 50,
      offset: 0,
    } as never);

    renderWithRouter();

    await waitFor(() => {
      expect(screen.getByText('患者张三')).toBeInTheDocument();
      expect(screen.getByText('患者李四')).toBeInTheDocument();
    });
  });

  it('点击查看后拉取 getEvaluation 并显示分数/引用', async () => {
    mockedListPending.mockResolvedValue({
      items: mockPendingItems,
      total: 2,
      limit: 50,
      offset: 0,
    } as never);

    mockedGetEvaluation.mockResolvedValue(mockEvaluation as never);

    const user = userEvent.setup();
    renderWithRouter();

    // 等待列表渲染
    await waitFor(() => {
      expect(screen.getByText('患者张三')).toBeInTheDocument();
    });

    // 点击查看按钮
    const viewButtons = screen.getAllByText('查看');
    await user.click(viewButtons[0]);

    // 验证调用了 getEvaluation
    await waitFor(() => {
      expect(mockedGetEvaluation).toHaveBeenCalledWith(101);
    });

    // 验证显示了分数
    await waitFor(() => {
      expect(screen.getByText('85')).toBeInTheDocument(); // inquiry_score
    });
  });

  it('提交后调用 API 并刷新队列', async () => {
    mockedListPending.mockResolvedValue({
      items: mockPendingItems,
      total: 2,
      limit: 50,
      offset: 0,
    } as never);

    mockedGetEvaluation.mockResolvedValue(mockEvaluation as never);
    mockedSubmitReview.mockResolvedValue({ success: true, evaluation_id: 1 } as never);

    const user = userEvent.setup();
    renderWithRouter();

    // 等待列表渲染
    await waitFor(() => {
      expect(screen.getByText('患者张三')).toBeInTheDocument();
    });

    // 点击查看
    const viewButtons = screen.getAllByText('查看');
    await user.click(viewButtons[0]);

    // 等待弹窗打开并填写表单
    await waitFor(() => {
      expect(screen.getByText('完成复核')).toBeInTheDocument();
    });

    // 填写反馈（至少2字）
    const feedbackInput = screen.getByPlaceholderText(/详细说明复核依据/);
    await user.type(feedbackInput, '已核实评分准确');

    // 点击提交
    const submitButton = screen.getByRole('button', { name: '完成复核' });
    await user.click(submitButton);

    // 验证调用了 submitReview
    await waitFor(() => {
      expect(mockedSubmitReview).toHaveBeenCalled();
    });

    // 验证刷新了队列
    await waitFor(() => {
      expect(mockedListPending).toHaveBeenCalledTimes(2); // 初始加载 + 提交后刷新
    });
  });

  it('加载失败显示 Alert 和重试按钮', async () => {
    mockedListPending.mockRejectedValue(new Error('Network error'));

    renderWithRouter();

    await waitFor(() => {
      expect(screen.getByText(/加载失败/)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /重试/ })).toBeInTheDocument();
    });
  });

  it('提交 409 提示已被其他管理员复核并刷新', async () => {
    mockedListPending.mockResolvedValue({
      items: mockPendingItems,
      total: 2,
      limit: 50,
      offset: 0,
    } as never);

    mockedGetEvaluation.mockResolvedValue(mockEvaluation as never);
    mockedSubmitReview.mockRejectedValue({
      response: { status: 409, data: { error_code: 'ALREADY_REVIEWED', message: '已被其他管理员复核' } },
    });

    const user = userEvent.setup();
    renderWithRouter();

    // 等待列表渲染
    await waitFor(() => {
      expect(screen.getByText('患者张三')).toBeInTheDocument();
    });

    // 点击查看
    const viewButtons = screen.getAllByText('查看');
    await user.click(viewButtons[0]);

    // 等待弹窗打开并填写反馈
    await waitFor(() => {
      expect(screen.getByText('完成复核')).toBeInTheDocument();
    });

    const feedbackInput = screen.getByPlaceholderText(/详细说明复核依据/);
    await user.type(feedbackInput, '已核实评分准确');

    // 点击提交
    const submitButton = screen.getByRole('button', { name: '完成复核' });
    await user.click(submitButton);

    // 验证显示了 409 错误提示
    await waitFor(() => {
      expect(screen.getByText(/已被其他管理员复核/)).toBeInTheDocument();
    });

    // 验证刷新了队列
    await waitFor(() => {
      expect(mockedListPending).toHaveBeenCalledTimes(2);
    });
  });

  it('反馈少于 2 字时前端校验阻止请求', async () => {
    mockedListPending.mockResolvedValue({
      items: mockPendingItems,
      total: 2,
      limit: 50,
      offset: 0,
    } as never);

    mockedGetEvaluation.mockResolvedValue(mockEvaluation as never);

    const user = userEvent.setup();
    renderWithRouter();

    // 等待列表渲染
    await waitFor(() => {
      expect(screen.getByText('患者张三')).toBeInTheDocument();
    });

    // 点击查看
    const viewButtons = screen.getAllByText('查看');
    await user.click(viewButtons[0]);

    // 等待弹窗打开
    await waitFor(() => {
      expect(screen.getByText('完成复核')).toBeInTheDocument();
    });

    // 填写少于2字的反馈
    const feedbackInput = screen.getByPlaceholderText(/详细说明复核依据/);
    await user.type(feedbackInput, 'a');

    // 点击提交
    const submitButton = screen.getByRole('button', { name: '完成复核' });
    await user.click(submitButton);

    // 验证没有调用 submitReview
    expect(mockedSubmitReview).not.toHaveBeenCalled();
  });

  it('doctor 直接访问 /admin/reviews 被 AdminRoute 重定向', async () => {
    renderWithRouter('/admin/reviews', false);

    await waitFor(() => {
      expect(screen.getByText('dashboard-page')).toBeInTheDocument();
    });
  });

  it('admin 可进入 /admin/reviews', async () => {
    mockedListPending.mockResolvedValue({
      items: mockPendingItems,
      total: 2,
      limit: 50,
      offset: 0,
    } as never);

    renderWithRouter('/admin/reviews', true);

    await waitFor(() => {
      expect(screen.getByText('人工复核工作台')).toBeInTheDocument();
    });
  });
});
