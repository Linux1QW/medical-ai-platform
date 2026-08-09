import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// ── Mock API module ──────────────────────────────────────────────────
vi.mock('../api/evaluation', () => ({
  createEvaluation: vi.fn(),
  getEvaluationRunStatus: vi.fn(),
  cancelEvaluationRun: vi.fn(),
}));

vi.mock('../utils/apiUrl', () => ({
  API_BASE_URL: '/api/v1',
  toWebSocketUrl: (path: string) => {
    if (path.startsWith('ws://') || path.startsWith('wss://')) return path;
    return `ws://localhost${path.startsWith('/') ? '' : '/'}${path}`;
  },
}));

// ── Fake WebSocket ───────────────────────────────────────────────────
class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  url: string;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  readyState = 0;
  sent: unknown[] = [];

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  send(data: string) {
    this.sent.push(JSON.parse(data));
  }

  close() {
    this.readyState = 3;
    this.onclose?.();
  }

  simulateOpen() {
    this.readyState = 1;
    this.onopen?.();
  }

  simulateMessage(data: Record<string, unknown>) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }

  simulateError() {
    this.onerror?.();
  }
}

// ── Import AFTER mocks ───────────────────────────────────────────────
import { createEvaluation, getEvaluationRunStatus, cancelEvaluationRun } from '../api/evaluation';
import { useEvaluationJob } from './useEvaluationJob';

const mockedCreate = createEvaluation as ReturnType<typeof vi.fn>;
const mockedGetStatus = getEvaluationRunStatus as ReturnType<typeof vi.fn>;
const mockedCancel = cancelEvaluationRun as ReturnType<typeof vi.fn>;

describe('useEvaluationJob', () => {
  beforeEach(() => {
    vi.stubGlobal('WebSocket', FakeWebSocket);
    FakeWebSocket.instances = [];
    vi.clearAllMocks();
    sessionStorage.setItem('token', 'test-jwt');
    // Default: prevent polling from terminating
    mockedGetStatus.mockResolvedValue({ status: 'running', progress: 50, message: '' });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    sessionStorage.clear();
  });

  const onReportReady = vi.fn().mockResolvedValue(undefined);

  it('202 正常流程：POST → WS 连接 → 轮询 completed → 回调', async () => {
    mockedCreate.mockResolvedValue({
      run_id: 'run-1',
      websocket_url: '/api/v1/evaluations/ws/runs/run-1',
      status: 'queued',
    });
    mockedGetStatus
      .mockResolvedValueOnce({ status: 'running', progress: 50, message: '分析中' })
      .mockResolvedValueOnce({ status: 'completed', progress: 100, message: '完成' });

    const { result } = renderHook(() => useEvaluationJob(1, onReportReady));

    await act(async () => {
      await result.current.start();
    });

    await waitFor(() => {
      expect(result.current.state.runId).toBe('run-1');
    }, { timeout: 3000 });

    expect(result.current.state.isActive).toBe(true);

    const ws = FakeWebSocket.instances[0];
    act(() => { ws.simulateOpen(); });
    expect(ws.sent[0]).toEqual({ type: 'auth', token: 'test-jwt' });

    act(() => {
      ws.simulateMessage({ type: 'progress', run_id: 'run-1', sequence: 1, progress: 30, message: 'RAG 检索' });
    });
    expect(result.current.state.progress).toBe(30);

    await waitFor(() => {
      expect(result.current.state.status).toBe('completed');
    }, { timeout: 15000 });

    expect(result.current.state.isActive).toBe(false);
    expect(onReportReady).toHaveBeenCalled();
  });

  it('409 恢复已有 run', async () => {
    const conflictError = {
      response: {
        status: 409,
        data: {
          error_code: 'EVALUATION_IN_PROGRESS',
          context: { run_id: 'existing-run', websocket_url: '/ws/runs/existing-run', status: 'running' },
        },
      },
    };
    mockedCreate.mockRejectedValueOnce(conflictError);
    mockedGetStatus.mockResolvedValueOnce({ status: 'completed', progress: 100, message: '完成' });

    const { result } = renderHook(() => useEvaluationJob(1, onReportReady));

    await act(async () => {
      await result.current.start();
    });

    await waitFor(() => {
      expect(result.current.state.runId).toBe('existing-run');
    }, { timeout: 3000 });

    expect(result.current.state.isActive).toBe(true);
  });

  it('latest progress 重放', async () => {
    mockedCreate.mockResolvedValue({
      run_id: 'run-2',
      websocket_url: '/ws/runs/run-2',
      status: 'running',
    });

    const { result } = renderHook(() => useEvaluationJob(2, onReportReady));

    await act(async () => {
      await result.current.start();
    });

    await waitFor(() => {
      expect(FakeWebSocket.instances.length).toBeGreaterThan(0);
    }, { timeout: 3000 });

    const ws = FakeWebSocket.instances[0];
    act(() => { ws.simulateOpen(); });
    act(() => {
      ws.simulateMessage({ type: 'latest', run_id: 'run-2', sequence: 5, progress: 75, message: '知识评估' });
    });

    expect(result.current.state.progress).toBe(75);
    expect(result.current.state.message).toBe('知识评估');
  });

  it('乱序 sequence 忽略', async () => {
    mockedCreate.mockResolvedValue({
      run_id: 'run-3',
      websocket_url: '/ws/runs/run-3',
      status: 'running',
    });

    const { result } = renderHook(() => useEvaluationJob(3, onReportReady));

    await act(async () => {
      await result.current.start();
    });

    await waitFor(() => {
      expect(FakeWebSocket.instances.length).toBeGreaterThan(0);
    }, { timeout: 3000 });

    const ws = FakeWebSocket.instances[0];
    act(() => { ws.simulateOpen(); });

    act(() => {
      ws.simulateMessage({ type: 'progress', run_id: 'run-3', sequence: 10, progress: 80, message: 'seq10' });
    });
    expect(result.current.state.progress).toBe(80);

    act(() => {
      ws.simulateMessage({ type: 'progress', run_id: 'run-3', sequence: 5, progress: 50, message: 'seq5' });
    });
    expect(result.current.state.progress).toBe(80);
  });

  it('WS 失败降级轮询', async () => {
    mockedCreate.mockResolvedValue({
      run_id: 'run-4',
      websocket_url: '/ws/runs/run-4',
      status: 'queued',
    });
    mockedGetStatus.mockResolvedValueOnce({ status: 'completed', progress: 100, message: '完成' });

    const { result } = renderHook(() => useEvaluationJob(4, onReportReady));

    await act(async () => {
      await result.current.start();
    });

    await waitFor(() => {
      expect(FakeWebSocket.instances.length).toBeGreaterThan(0);
    }, { timeout: 3000 });

    const ws = FakeWebSocket.instances[0];
    act(() => { ws.simulateOpen(); });
    act(() => { ws.simulateError(); });

    await waitFor(() => {
      expect(result.current.state.status).toBe('completed');
    }, { timeout: 15000 });

    expect(onReportReady).toHaveBeenCalled();
  });

  it('取消流程', async () => {
    mockedCreate.mockResolvedValue({
      run_id: 'run-5',
      websocket_url: '/ws/runs/run-5',
      status: 'queued',
    });
    mockedCancel.mockResolvedValue({ status: 'cancelled', cancel_requested: false });
    // Prevent polling from interfering
    mockedGetStatus.mockRejectedValue(new Error('should not poll'));

    const { result } = renderHook(() => useEvaluationJob(5, onReportReady));

    await act(async () => {
      await result.current.start();
    });

    await waitFor(() => {
      expect(result.current.state.runId).toBe('run-5');
    }, { timeout: 3000 });

    expect(result.current.state.isActive).toBe(true);

    await act(async () => {
      await result.current.cancel();
    });

    // cancel returns cancelled directly
    expect(result.current.state.status).toBe('cancelled');
    expect(result.current.state.isActive).toBe(false);
  }, 20000);

  it('unmount 清理 socket 和 timer', async () => {
    mockedCreate.mockResolvedValue({
      run_id: 'run-6',
      websocket_url: '/ws/runs/run-6',
      status: 'queued',
    });

    const { result, unmount } = renderHook(() => useEvaluationJob(6, onReportReady));

    await act(async () => {
      await result.current.start();
    });

    await waitFor(() => {
      expect(FakeWebSocket.instances.length).toBeGreaterThan(0);
    }, { timeout: 3000 });

    const ws = FakeWebSocket.instances[0];
    act(() => { ws.simulateOpen(); });
    const closeSpy = vi.spyOn(ws, 'close');

    unmount();
    expect(closeSpy).toHaveBeenCalled();
  });
});
