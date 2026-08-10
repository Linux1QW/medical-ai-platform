export interface CoachStreamRequest {
  consultation_id: number;
  doctor_id: number;
  latest_message: string;
  idempotency_key: string;
}

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

const API_BASE = '/api/v1/coach';

export async function getCoachState(consultationId: number): Promise<CoachState> {
  const res = await fetch(`${API_BASE}/consultations/${consultationId}/state`);
  if (!res.ok) throw new Error(`Failed to get coach state: ${res.status}`);
  return res.json();
}

export async function submitFeedback(
  suggestionId: string,
  feedback: FeedbackRequest,
): Promise<{ suggestion_id: string; feedback: string; recorded: boolean }> {
  const res = await fetch(`${API_BASE}/suggestions/${suggestionId}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(feedback),
  });
  if (!res.ok) throw new Error(`Failed to submit feedback: ${res.status}`);
  return res.json();
}

export function createCoachSSEStream(
  consultationId: number,
  request: CoachStreamRequest,
  onEvent: (event: string, data: Record<string, unknown>) => void,
  lastEventId?: string,
): AbortController {
  const controller = new AbortController();

  fetch(`${API_BASE}/consultations/${consultationId}/suggestions/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(lastEventId ? { 'Last-Event-ID': lastEventId } : {}),
    },
    body: JSON.stringify(request),
    signal: controller.signal,
  })
    .then(async (response) => {
      const reader = response.body?.getReader();
      if (!reader) return;

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEvent = '';
        let currentData = '';
        let currentId = '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7);
          } else if (line.startsWith('data: ')) {
            currentData = line.slice(6);
          } else if (line.startsWith('id: ')) {
            currentId = line.slice(4);
          } else if (line === '' && currentEvent) {
            try {
              const data = JSON.parse(currentData) as Record<string, unknown>;
              onEvent(currentEvent, { ...data, _id: currentId || undefined });
            } catch {
              onEvent(currentEvent, { _raw: currentData, _id: currentId || undefined });
            }
            currentEvent = '';
            currentData = '';
            currentId = '';
          }
        }
      }
    })
    .catch((err: Error) => {
      if (err.name !== 'AbortError') {
        onEvent('error', { message: err.message });
      }
    });

  return controller;
}
