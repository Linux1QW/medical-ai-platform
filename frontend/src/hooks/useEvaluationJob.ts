import { useCallback, useEffect, useRef, useState } from 'react';
import { createEvaluation, getEvaluationRunStatus, cancelEvaluationRun } from '../api/evaluation';
import { toWebSocketUrl } from '../utils/apiUrl';
import type { EvaluationJobStatus, EvaluationProgressEvent } from '../types';

export type EvaluationJobState = {
  runId: string | null;
  status: EvaluationJobStatus | 'idle';
  progress: number;
  message: string;
  errorCode: string | null;
  cancelRequested: boolean;
  isActive: boolean;
};

const TERMINAL_STATUSES: EvaluationJobStatus[] = ['completed', 'reviewed', 'needs_review', 'failed', 'cancelled'];
const SUCCESS_STATUSES: EvaluationJobStatus[] = ['completed', 'reviewed', 'needs_review'];

const initialState: EvaluationJobState = {
  runId: null,
  status: 'idle',
  progress: 0,
  message: '',
  errorCode: null,
  cancelRequested: false,
  isActive: false,
};

export function useEvaluationJob(
  consultationId: number,
  onReportReady: () => Promise<void>,
): {
  state: EvaluationJobState;
  start: () => Promise<void>;
  cancel: () => Promise<void>;
  resume: () => Promise<void>;
} {
  const [state, setState] = useState<EvaluationJobState>(initialState);
  const wsRef = useRef<WebSocket | null>(null);
  const pollTimerRef = useRef<number | null>(null);
  const sequenceRef = useRef<number>(0);
  const wsHealthyRef = useRef<boolean>(false);
  const mountedRef = useRef<boolean>(true);
  const onReportReadyRef = useRef(onReportReady);
  onReportReadyRef.current = onReportReady;

  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      cleanup();
    };
  }, [cleanup]);

  const startPolling = useCallback((runId: string) => {
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);

    const poll = async () => {
      try {
        const runStatus = await getEvaluationRunStatus(runId);
        if (!mountedRef.current) return;

        setState((prev) => ({
          ...prev,
          status: runStatus.status,
          progress: runStatus.progress,
          message: runStatus.message,
          cancelRequested: runStatus.cancel_requested ?? prev.cancelRequested,
          errorCode: runStatus.error_code ?? null,
          isActive: !TERMINAL_STATUSES.includes(runStatus.status),
        }));

        if (TERMINAL_STATUSES.includes(runStatus.status)) {
          cleanup();
          if (SUCCESS_STATUSES.includes(runStatus.status)) {
            await onReportReadyRef.current();
          }
        } else {
          // Adjust polling interval based on WS health
          const interval = wsHealthyRef.current ? 5000 : 2000;
          if (pollTimerRef.current) clearInterval(pollTimerRef.current);
          pollTimerRef.current = window.setTimeout(poll, interval) as unknown as number;
        }
      } catch {
        if (!mountedRef.current) return;
        // Network error: exponential backoff, max 10s
        if (pollTimerRef.current) clearInterval(pollTimerRef.current);
        pollTimerRef.current = window.setTimeout(poll, Math.min(10000, 2000 * 2)) as unknown as number;
      }
    };

    pollTimerRef.current = window.setTimeout(poll, 2000) as unknown as number;
  }, [cleanup]);

  const connectWs = useCallback((wsUrl: string, runId: string) => {
    const token = sessionStorage.getItem('token') || '';
    const url = toWebSocketUrl(wsUrl);
    const ws = new WebSocket(url);
    wsRef.current = ws;
    wsHealthyRef.current = false;

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'auth', token }));
    };

    ws.onmessage = (event) => {
      try {
        const data: EvaluationProgressEvent = JSON.parse(event.data);

        if (data.type === 'auth_ok') {
          wsHealthyRef.current = true;
          return;
        }

        // Only process events for our run_id
        if (data.run_id && data.run_id !== runId) return;

        // Sequence check: ignore stale events
        if (data.sequence !== undefined) {
          if (data.sequence <= sequenceRef.current) return;
          sequenceRef.current = data.sequence;
        }

        if (data.type === 'progress' || data.type === 'latest') {
          setState((prev) => ({
            ...prev,
            progress: data.progress,
            message: data.message,
          }));
        } else if (data.type === 'error') {
          setState((prev) => ({
            ...prev,
            errorCode: data.error_code ?? null,
            message: data.message,
          }));
        } else if (data.type === 'done') {
          // Server signals completion
          wsHealthyRef.current = false;
        }
      } catch {
        // ignore parse errors
      }
    };

    ws.onerror = () => {
      wsHealthyRef.current = false;
    };

    ws.onclose = () => {
      wsHealthyRef.current = false;
      wsRef.current = null;
    };
  }, []);

  const start = useCallback(async () => {
    setState({ ...initialState, status: 'queued', isActive: true, message: '正在初始化评估环境...' });
    sequenceRef.current = 0;

    try {
      const data = await createEvaluation(consultationId);
      if (!mountedRef.current) return;

      const runId = data.run_id;
      setState((prev) => ({
        ...prev,
        runId,
        status: data.status,
      }));

      // Connect WS and start polling
      connectWs(data.websocket_url, runId);
      startPolling(runId);
    } catch (error: unknown) {
      if (!mountedRef.current) return;

      // 409 Conflict: recover existing run
      const errResp = (error as { response?: { status?: number; data?: { error_code?: string; context?: { run_id?: string; websocket_url?: string; status?: EvaluationJobStatus } } } })?.response;
      if (errResp?.status === 409 && errResp.data?.context?.run_id) {
        const ctx = errResp.data.context;
        const runId = ctx.run_id as string;
        setState((prev) => ({
          ...prev,
          runId,
          status: ctx.status ?? 'running',
          isActive: true,
        }));

        if (ctx.websocket_url) {
          connectWs(ctx.websocket_url, runId);
        }
        startPolling(runId);
      } else {
        setState((prev) => ({
          ...prev,
          status: 'failed',
          isActive: false,
          errorCode: (error as { response?: { data?: { error_code?: string } } })?.response?.data?.error_code ?? null,
          message: '创建评估失败',
        }));
      }
    }
  }, [consultationId, connectWs, startPolling]);

  const cancel = useCallback(async () => {
    const runId = state.runId;
    if (!runId) return;

    try {
      const resp = await cancelEvaluationRun(runId);
      if (!mountedRef.current) return;

      // If backend says still running + cancel_requested, keep polling
      if (resp.status === 'running' || resp.cancel_requested) {
        setState((prev) => ({
          ...prev,
          cancelRequested: true,
          message: '正在取消...',
        }));
      } else {
        setState((prev) => ({
          ...prev,
          status: 'cancelled' as EvaluationJobStatus,
          isActive: false,
          cancelRequested: false,
        }));
        cleanup();
      }
    } catch {
      // Cancel request failed, keep polling
    }
  }, [state.runId, cleanup]);

  const resume = useCallback(async () => {
    // Resume by checking lock status externally; caller provides runId via state
    // This is used when page loads and lock shows active run
    setState((prev) => ({
      ...prev,
      isActive: true,
      status: 'running',
      message: '恢复评估进度...',
    }));
  }, []);

  return { state, start, cancel, resume };
}
