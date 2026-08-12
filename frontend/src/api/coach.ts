import request from '../utils/request';
import { createSSEParser } from './sseParser';

// ── Types ────────────────────────────────────────────────────────────────────

export interface CoachState {
  consultation_id: number;
  status: 'disabled' | 'idle' | 'thinking' | 'suggestion' | 'degraded' | 'error';
  current_stage: string | null;
  turn_no: number;
}

export interface CoachSuggestion {
  suggestion_id: string;
  session_id: string;
  turn_no: number;
  intent: string;
  stage: string;
  suggested_question: string;
  rationale_summary: string;
  confidence: number;
  risk_level: 'low' | 'medium' | 'high';
  targeted_slots: string[];
  citation_ids: string[];
}

export interface FeedbackRequest {
  feedback: 'accepted' | 'rejected' | 'ignored';
  reason?: string;
}

export interface CoachStreamRequest {
  consultation_id: number;
  latest_message: string;
  idempotency_key: string;
}

export type CoachSSEEvent =
  | { type: 'thinking'; turn_no: number; stage: string }
  | { type: 'suggestion'; suggestion: CoachSuggestion }
  | { type: 'error'; message: string }
  | { type: 'done' };

export interface CoachTraceNode {
  node: string;
  status: 'started' | 'completed' | 'error';
  duration_ms: number | null;
  /** Sanitised, user-safe description. Never raw prompts or tool args. */
  safe_message: string | null;
}

export interface CoachTraceSession {
  session_id: string;
  consultation_id: number;
  status: string;
  created_at: string | null;
}

export interface CoachTraceDecision {
  decision_id: string;
  agent: string;
  action: string;
  rationale: string | null;
  timestamp: string | null;
}

export interface CoachTraceEvent {
  event_id: string;
  agent: string;
  event_type: string;
  message: string | null;
  timestamp: string | null;
}

export interface CoachTraceResponse {
  session: CoachTraceSession | null;
  decisions: CoachTraceDecision[];
  events: CoachTraceEvent[];
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function getAuthHeaders(): Record<string, string> {
  const token = sessionStorage.getItem('token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}

const API_BASE = '/coach';

// ── JSON endpoints (use authenticated axios instance) ─────────────────────────

/** Fetch current coach state for a consultation. */
export async function getCoachState(consultationId: number): Promise<CoachState> {
  return request.get(`${API_BASE}/consultations/${consultationId}/state`);
}

/** Submit user feedback on a suggestion. */
export async function submitFeedback(
  suggestionId: string,
  feedback: FeedbackRequest,
): Promise<{ suggestion_id: string; feedback: string; recorded: boolean }> {
  return request.post(`${API_BASE}/suggestions/${suggestionId}/feedback`, feedback);
}

/** Fetch sanitised agent trace projection (admin only). */
export async function getCoachTrace(consultationId: number): Promise<CoachTraceResponse> {
  return request.get(
    `/admin/consultations/${consultationId}/trace`,
  ) as Promise<CoachTraceResponse>;
}

// ── SSE stream (uses fetch with auth token) ──────────────────────────────────

export interface CoachStreamCallbacks {
  onEvent: (event: CoachSSEEvent, id?: string) => void;
  onError?: (error: Error) => void;
}

/**
 * Open a coach suggestion SSE stream.
 *
 * - Uses the session token for authentication (no doctorId parameter).
 * - Rejects non-2xx responses before reading the stream body.
 * - Supports `Last-Event-ID` for replay after reconnect.
 *
 * Returns an `AbortController` so the caller can cancel the stream.
 */
export function createCoachSSEStream(
  consultationId: number,
  streamRequest: CoachStreamRequest,
  callbacks: CoachStreamCallbacks,
  lastEventId?: string,
): AbortController {
  const controller = new AbortController();
  const parser = createSSEParser();

  const run = async () => {
    const response = await fetch(`/api/v1${API_BASE}/consultations/${consultationId}/suggestions/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        ...getAuthHeaders(),
        ...(lastEventId ? { 'Last-Event-ID': lastEventId } : {}),
      },
      body: JSON.stringify(streamRequest),
      signal: controller.signal,
    });

    if (!response.ok) {
      const body = await response.json().catch(() => null);
      const msg =
        (body as Record<string, string> | null)?.message ||
        (body as Record<string, string> | null)?.detail ||
        `HTTP ${response.status}`;
      throw new Error(msg);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('Unable to read response stream');

    const decoder = new TextDecoder();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const events = parser.push(chunk);

        for (const evt of events) {
          handleSSEEvent(evt.event, evt.data, callbacks, evt.id);
        }
      }
      // Flush any remaining event
      for (const evt of parser.flush()) {
        handleSSEEvent(evt.event, evt.data, callbacks, evt.id);
      }
    } finally {
      reader.releaseLock();
    }
  };

  run().catch((err: Error) => {
    if (err.name !== 'AbortError') {
      callbacks.onError?.(err);
    }
  });

  return controller;
}

function handleSSEEvent(
  eventType: string,
  rawData: unknown,
  callbacks: CoachStreamCallbacks,
  eventId?: string,
): void {
  const data = (rawData ?? {}) as Record<string, unknown>;

  switch (eventType) {
    case 'thinking':
      callbacks.onEvent({
        type: 'thinking',
        turn_no: (data.turn_no as number) ?? 0,
        stage: (data.stage as string) ?? '',
      }, eventId);
      break;

    case 'suggestion':
      callbacks.onEvent({
        type: 'suggestion',
        suggestion: data as unknown as CoachSuggestion,
      }, eventId);
      break;

    case 'error':
      callbacks.onEvent({
        type: 'error',
        message: (data.message as string) ?? 'Unknown error',
      }, eventId);
      break;

    case 'done':
      callbacks.onEvent({ type: 'done' }, eventId);
      break;

    default:
      // Unknown event type — ignore
      break;
  }
}
