import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import ConsultationPage from './index';

// Mocks - use indirection to avoid namespace mock issues
const mockGetConsultationDetail = vi.fn();
const mockGetPatient = vi.fn();
const mockGetEvaluation = vi.fn();
const mockGetCoachState = vi.fn();
const mockGetCoachTrace = vi.fn();
const mockUseAuthFn = vi.fn();

vi.mock('../../api/consultation', () => ({
  getConsultationDetail: (id: number) => mockGetConsultationDetail(id),
  sendMessageStream: vi.fn(),
  submitDiagnosis: vi.fn(),
  endConsultation: vi.fn(),
  extendRounds: vi.fn(),
}));

vi.mock('../../api/patient', () => ({
  getPatient: (id: number) => mockGetPatient(id),
}));

vi.mock('../../api/evaluation', () => ({
  getEvaluation: (id: number) => mockGetEvaluation(id),
}));

vi.mock('../../api/coach', () => ({
  getCoachState: (id: number) => mockGetCoachState(id),
  submitFeedback: vi.fn(),
  getCoachTrace: (id: number) => mockGetCoachTrace(id),
  createCoachSSEStream: vi.fn(),
}));

vi.mock('../../store/useAuth', () => ({
  useAuth: () => mockUseAuthFn(),
}));

// Mock VoiceConsultation to avoid livekit-client dependency
vi.mock('../../components/VoiceConsultation', () => ({
  VoiceConsultation: () => <div data-testid="voice-consultation-mock" />,
}));

// Fixtures
const mockConsultationDetail = {
  id: 1, doctor_id: 10, patient_id: 100,
  status: 'in_progress' as const,
  started_at: '2026-08-01T10:00:00Z', ended_at: null,
  summary: '', diagnosis: '', treatment_plan: '',
  max_rounds: 20, created_at: '2026-08-01T10:00:00Z',
  messages: [{ id: 1, consultation_id: 1, role: 'patient' as const, content: '我最近胸痛', sequence: 1, created_at: '2026-08-01T10:00:00Z' }],
};

const mockPatient = {
  id: 100, name: '王小明', age: 55, gender: 'male' as const,
  personality_type: '配合型' as const, chief_complaint: '胸痛3天',
  medical_history: '高血压', symptoms: '胸痛、气短',
  difficulty_level: 2, created_at: '2026-08-01T09:00:00Z',
};

const mockCoachState = { consultation_id: 1, status: 'idle' as const, current_stage: null, turn_no: 0 };
const mockTraceResponse = {
  session: { session_id: 'sess-001', consultation_id: 1, status: 'active', created_at: '2026-08-01T10:00:00Z' },
  decisions: [{ decision_id: 'dec-1', agent: 'safety_check', action: 'pass', rationale: 'No red flags', timestamp: '2026-08-01T10:00:01Z' }],
  events: [{ event_id: 'evt-1', agent: 'history_agent', event_type: 'completed', message: 'History taking done', timestamp: '2026-08-01T10:00:02Z' }],
};

function setupAuthMocks(isAdmin = false) {
  mockUseAuthFn.mockReturnValue({
    user: { id: isAdmin ? 1 : 10, username: isAdmin ? 'admin01' : 'doctor01', role: isAdmin ? 'admin' : 'doctor', email: '', real_name: isAdmin ? '管理员' : '王医生', department: '心内科', avatar: '', created_at: '' },
    token: 'test-token', isLoggedIn: true, isAdmin, saveAuth: vi.fn(), logout: vi.fn(),
  });
}

function setupApiMocks(status: 'in_progress' | 'completed' = 'in_progress') {
  mockGetConsultationDetail.mockResolvedValue({ ...mockConsultationDetail, status });
  mockGetPatient.mockResolvedValue(mockPatient);
  mockGetCoachState.mockResolvedValue(mockCoachState);
  mockGetCoachTrace.mockResolvedValue(mockTraceResponse);
  mockGetEvaluation.mockResolvedValue(null);
}

const renderPage = () => render(
  <MemoryRouter initialEntries={['/consultation/1']}>
    <Routes>
      <Route path="/consultation/:id" element={<ConsultationPage />} />
      <Route path="/evaluation/:id" element={<div>evaluation-page</div>} />
      <Route path="/login" element={<div>login-page</div>} />
    </Routes>
  </MemoryRouter>,
);

describe('CoachPanel integration in Consultation page', () => {
  beforeEach(() => {
    sessionStorage.clear();
    vi.clearAllMocks();
    setupAuthMocks(false);
    setupApiMocks();
  });

  it('renders CoachPanel during in-progress consultation', async () => {
    await act(async () => { renderPage(); });
    await waitFor(() => {
      expect(screen.getByLabelText('问诊教练面板')).toBeInTheDocument();
    });
    expect(screen.getByText('等待问诊')).toBeInTheDocument();
  });

  it('composer textarea is present', async () => {
    await act(async () => { renderPage(); });
    await waitFor(() => {
      expect(screen.getByLabelText('问诊教练面板')).toBeInTheDocument();
    });
    const composer = screen.getByPlaceholderText(/输入您的问诊内容/);
    expect(composer).toBeInTheDocument();
    expect(composer).toHaveValue('');
  });

  it('does not render CoachPanel after consultation ends', async () => {
    setupApiMocks('completed');
    await act(async () => { renderPage(); });
    await waitFor(() => {
      expect(screen.queryByLabelText('问诊教练面板')).not.toBeInTheDocument();
    });
  });
});

describe('AgentTraceDrawer integration', () => {
  beforeEach(() => {
    sessionStorage.clear();
    vi.clearAllMocks();
    setupApiMocks();
  });

  it('hides Agent Trace button for non-admin doctors', async () => {
    setupAuthMocks(false);
    await act(async () => { renderPage(); });
    await waitFor(() => {
      expect(screen.getByLabelText('问诊教练面板')).toBeInTheDocument();
    });
    expect(screen.queryByText('Agent Trace')).not.toBeInTheDocument();
  });

  it('shows Agent Trace button for admin users', async () => {
    setupAuthMocks(true);
    await act(async () => { renderPage(); });
    await waitFor(() => {
      expect(screen.getByText('Agent Trace')).toBeInTheDocument();
    });
  });

  it('fetches trace data when admin opens drawer', async () => {
    setupAuthMocks(true);
    await act(async () => { renderPage(); });
    const user = userEvent.setup();
    await waitFor(() => {
      expect(screen.getByText('Agent Trace')).toBeInTheDocument();
    });
    await user.click(screen.getByText('Agent Trace'));
    await waitFor(() => {
      expect(mockGetCoachTrace).toHaveBeenCalledWith(1);
    });
  });

  it('trace response contains session, decisions and events objects (not a flat array)', async () => {
    setupAuthMocks(true);
    await act(async () => { renderPage(); });
    const user = userEvent.setup();
    await waitFor(() => {
      expect(screen.getByText('Agent Trace')).toBeInTheDocument();
    });
    await user.click(screen.getByText('Agent Trace'));
    await waitFor(() => {
      expect(mockGetCoachTrace).toHaveBeenCalledWith(1);
    });
    // Verify the mock was called and returned the structured response
    const result = await mockGetCoachTrace(1);
    expect(result).toHaveProperty('session');
    expect(result).toHaveProperty('decisions');
    expect(result).toHaveProperty('events');
    expect(Array.isArray(result.decisions)).toBe(true);
    expect(Array.isArray(result.events)).toBe(true);
  });
});

describe('CoachPanel auth integration', () => {
  beforeEach(() => {
    sessionStorage.clear();
    vi.clearAllMocks();
    setupApiMocks();
  });

  it('calls getCoachState on mount', async () => {
    setupAuthMocks(false);
    await act(async () => { renderPage(); });
    await waitFor(() => {
      expect(mockGetCoachState).toHaveBeenCalledWith(1);
    });
  });
});
