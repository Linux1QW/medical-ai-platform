import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { CoachPanel } from '../components/CoachPanel';
import type { CoachSuggestion } from '../api/coach';

// Mock the hook
vi.mock('../hooks/useCoachSuggestion', () => ({
  useCoachSuggestion: vi.fn(),
}));

import { useCoachSuggestion } from '../hooks/useCoachSuggestion';

const mockHook = useCoachSuggestion as ReturnType<typeof vi.fn>;

const mockSuggestion: CoachSuggestion = {
  suggestion_id: 'sug-1',
  session_id: 'sess-1',
  turn_no: 1,
  intent: 'clarify_symptom',
  stage: 'history-taking',
  suggested_question: '您是否有胸闷的症状？',
  rationale_summary: '患者提到胸痛，需进一步排查胸闷',
  confidence: 0.85,
  risk_level: 'low',
  targeted_slots: ['chest_tightness'],
  citation_ids: ['cite-1'],
};

function setupHook(overrides: Record<string, unknown> = {}) {
  mockHook.mockReturnValue({
    status: 'idle',
    suggestion: null,
    errorMessage: null,
    turnNo: 0,
    history: [],
    requestSuggestion: vi.fn(),
    retryLastSuggestion: vi.fn(),
    handleFeedback: vi.fn(),
    dismissSuggestion: vi.fn(),
    ...overrides,
  });
}

describe('CoachPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows "教练已关闭" when status is disabled', () => {
    setupHook({ status: 'disabled' });
    render(<CoachPanel consultationId={1} onApplySuggestion={vi.fn()} />);
    expect(screen.getByText('教练已关闭')).toBeInTheDocument();
    expect(screen.getByText('教练功能已关闭。请联系管理员启用。')).toBeInTheDocument();
  });

  it('shows textarea and button when status is idle', () => {
    setupHook({ status: 'idle' });
    render(<CoachPanel consultationId={1} onApplySuggestion={vi.fn()} />);
    expect(screen.getByText('等待问诊')).toBeInTheDocument();
    expect(screen.getByLabelText('教练输入框')).toBeInTheDocument();
    expect(screen.getByText('给我一个提示')).toBeInTheDocument();
  });

  it('displays suggestion text when status is suggestion', () => {
    setupHook({ status: 'suggestion', suggestion: mockSuggestion });
    render(<CoachPanel consultationId={1} onApplySuggestion={vi.fn()} />);
    expect(screen.getByText('建议就绪')).toBeInTheDocument();
    expect(screen.getByText('建议问题：')).toBeInTheDocument();
    expect(screen.getByText('您是否有胸闷的症状？')).toBeInTheDocument();
    expect(screen.getByText('患者提到胸痛，需进一步排查胸闷')).toBeInTheDocument();
    expect(screen.getByText('置信度: 85%')).toBeInTheDocument();
  });

  it('calls onApplySuggestion with suggested text when "填入输入框" is clicked', () => {
    const onApplySuggestion = vi.fn();
    setupHook({
      status: 'suggestion',
      suggestion: mockSuggestion,
    });
    render(<CoachPanel consultationId={1} onApplySuggestion={onApplySuggestion} />);
    fireEvent.click(screen.getByText('填入输入框'));
    expect(onApplySuggestion).toHaveBeenCalledWith('您是否有胸闷的症状？');
  });

  it('shows thinking indicator when status is thinking', () => {
    setupHook({ status: 'thinking' });
    render(<CoachPanel consultationId={1} onApplySuggestion={vi.fn()} />);
    expect(screen.getByText('正在分析...')).toBeInTheDocument();
    expect(screen.getByText('正在分析您的问诊内容...')).toBeInTheDocument();
  });

  it('has stable test ID on request button', () => {
    setupHook({ status: 'idle' });
    render(<CoachPanel consultationId={1} onApplySuggestion={vi.fn()} />);
    expect(screen.getByTestId('coach-request-btn')).toBeInTheDocument();
  });

  it('has stable test IDs on accept/reject buttons when suggestion is shown', () => {
    setupHook({ status: 'suggestion', suggestion: mockSuggestion });
    render(<CoachPanel consultationId={1} onApplySuggestion={vi.fn()} />);
    expect(screen.getByTestId('coach-accept-btn')).toBeInTheDocument();
    expect(screen.getByTestId('coach-reject-btn')).toBeInTheDocument();
  });
});
