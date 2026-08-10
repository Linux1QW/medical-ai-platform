import { useState, useCallback, useRef, useEffect } from 'react';
import type { CoachSuggestion, FeedbackRequest, CoachSSEEvent } from '../api/coach';
import { createCoachSSEStream, getCoachState, submitFeedback } from '../api/coach';

export type CoachPanelStatus = 'disabled' | 'idle' | 'thinking' | 'suggestion' | 'degraded' | 'error';

interface UseCoachSuggestionReturn {
  status: CoachPanelStatus;
  suggestion: CoachSuggestion | null;
  errorMessage: string | null;
  turnNo: number;
  history: CoachSuggestion[];
  requestSuggestion: (message: string) => void;
  retryLastSuggestion: () => void;
  handleFeedback: (feedback: FeedbackRequest) => Promise<void>;
  dismissSuggestion: () => void;
}

/**
 * Generate a cryptographic UUID v4 idempotency key.
 * Uses `crypto.randomUUID()` when available, otherwise falls back to
 * `crypto.getRandomValues()` to build a RFC-4122 compliant UUID.
 */
function generateIdempotencyKey(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  // Fallback for environments without randomUUID
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  // Set version 4
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  // Set variant 10xx
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export function useCoachSuggestion(consultationId: number): UseCoachSuggestionReturn {
  const [status, setStatus] = useState<CoachPanelStatus>('idle');
  const [suggestion, setSuggestion] = useState<CoachSuggestion | null>(null);
  const [errorMessage, setError] = useState<string | null>(null);
  const [turnNo, setTurnNo] = useState(0);
  const [history, setHistory] = useState<CoachSuggestion[]>([]);

  const controllerRef = useRef<AbortController | null>(null);
  const lastEventIdRef = useRef<string | undefined>(undefined);
  const turnNoRef = useRef(turnNo);

  /**
   * Idempotency key for the current in-flight request.
   * Retained until the stream reaches `done` so retries use the same key.
   */
  const idempotencyKeyRef = useRef<string | null>(null);

  /** The latest message that was sent to the coach (for retry). */
  const lastMessageRef = useRef<string>('');

  // Keep ref in sync so SSE callback always sees latest value
  useEffect(() => {
    turnNoRef.current = turnNo;
  }, [turnNo]);

  // Check initial state on mount / consultation change
  useEffect(() => {
    getCoachState(consultationId)
      .then((state) => setStatus(state.status))
      .catch(() => setStatus('error'));
  }, [consultationId]);

  const handleSSEEvent = useCallback((event: CoachSSEEvent) => {
    switch (event.type) {
      case 'thinking':
        setStatus('thinking');
        setTurnNo(event.turn_no || turnNoRef.current + 1);
        break;

      case 'suggestion':
        setStatus('suggestion');
        setSuggestion(event.suggestion);
        break;

      case 'error':
        setStatus('degraded');
        setError(event.message);
        break;

      case 'done':
        // Stream finished — release the idempotency key
        idempotencyKeyRef.current = null;
        break;
    }
  }, []);

  const requestSuggestion = useCallback((message: string) => {
    // Abort previous stream if any
    controllerRef.current?.abort();

    // Generate a new idempotency key only if there is no in-flight request
    if (!idempotencyKeyRef.current) {
      idempotencyKeyRef.current = generateIdempotencyKey();
    }

    lastMessageRef.current = message;
    setStatus('thinking');
    setError(null);

    const controller = createCoachSSEStream(
      consultationId,
      {
        consultation_id: consultationId,
        latest_message: message,
        idempotency_key: idempotencyKeyRef.current,
      },
      {
        onEvent: handleSSEEvent,
        onError: (err) => {
          setStatus('degraded');
          setError(err.message || 'Connection failed');
        },
      },
      lastEventIdRef.current,
    );

    controllerRef.current = controller;
  }, [consultationId, handleSSEEvent]);

  /** Retry the last request with the same idempotency key (if still held). */
  const retryLastSuggestion = useCallback(() => {
    if (lastMessageRef.current) {
      requestSuggestion(lastMessageRef.current);
    }
  }, [requestSuggestion]);

  const handleFeedback = useCallback(async (feedback: FeedbackRequest) => {
    if (!suggestion) return;
    try {
      await submitFeedback(suggestion.suggestion_id, feedback);
      // After feedback, return to idle and retain suggestion in history
      setHistory((prev) => [...prev, suggestion]);
      setSuggestion(null);
      setStatus('idle');
    } catch {
      setError('Failed to submit feedback');
    }
  }, [suggestion]);

  /** Dismiss the current suggestion without recording feedback. */
  const dismissSuggestion = useCallback(() => {
    if (suggestion) {
      setHistory((prev) => [...prev, suggestion]);
    }
    setSuggestion(null);
    setStatus('idle');
  }, [suggestion]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
    };
  }, []);

  return {
    status,
    suggestion,
    errorMessage,
    turnNo,
    history,
    requestSuggestion,
    retryLastSuggestion,
    handleFeedback,
    dismissSuggestion,
  };
}
