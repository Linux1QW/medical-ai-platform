import { useState, useCallback, useRef, useEffect } from 'react';
import type { CoachSuggestion, FeedbackRequest } from '../api/coach';
import { createCoachSSEStream, getCoachState, submitFeedback } from '../api/coach';

export type CoachPanelStatus = 'disabled' | 'idle' | 'thinking' | 'suggestion' | 'degraded' | 'error';

interface UseCoachSuggestionReturn {
  status: CoachPanelStatus;
  suggestion: CoachSuggestion | null;
  errorMessage: string | null;
  turnNo: number;
  requestSuggestion: (message: string) => void;
  handleFeedback: (feedback: FeedbackRequest) => Promise<void>;
  pendingText: string;
  setPendingText: (text: string) => void;
  applyToInput: () => string | null;
}

export function useCoachSuggestion(
  consultationId: number,
  doctorId: number,
): UseCoachSuggestionReturn {
  const [status, setStatus] = useState<CoachPanelStatus>('idle');
  const [suggestion, setSuggestion] = useState<CoachSuggestion | null>(null);
  const [errorMessage, setError] = useState<string | null>(null);
  const [turnNo, setTurnNo] = useState(0);
  const [pendingText, setPendingText] = useState('');
  const controllerRef = useRef<AbortController | null>(null);
  const lastEventIdRef = useRef<string | undefined>(undefined);
  const turnNoRef = useRef(turnNo);

  // Keep ref in sync so SSE callback always sees latest value
  useEffect(() => {
    turnNoRef.current = turnNo;
  }, [turnNo]);

  // Check initial state
  useEffect(() => {
    getCoachState(consultationId)
      .then((state) => setStatus(state.status))
      .catch(() => setStatus('error'));
  }, [consultationId]);

  const requestSuggestion = useCallback((message: string) => {
    // Abort previous stream
    controllerRef.current?.abort();

    const idempotencyKey = `${consultationId}-${Date.now()}`;

    setStatus('thinking');
    setError(null);

    const controller = createCoachSSEStream(
      consultationId,
      {
        consultation_id: consultationId,
        doctor_id: doctorId,
        latest_message: message,
        idempotency_key: idempotencyKey,
      },
      (event, data) => {
        if (data._id) lastEventIdRef.current = data._id as string;

        switch (event) {
          case 'thinking':
            setStatus('thinking');
            setTurnNo((data.turn_no as number) || turnNoRef.current + 1);
            break;
          case 'suggestion':
            setStatus('suggestion');
            setSuggestion(data as unknown as CoachSuggestion);
            break;
          case 'error':
            setStatus('degraded');
            setError((data.message as string) || 'Unknown error');
            break;
          case 'done':
            // Stream complete
            break;
        }
      },
      lastEventIdRef.current,
    );

    controllerRef.current = controller;
  }, [consultationId, doctorId]);

  const handleFeedback = useCallback(async (feedback: FeedbackRequest) => {
    if (!suggestion) return;
    try {
      await submitFeedback(suggestion.suggestion_id, feedback);
    } catch {
      setError('Failed to submit feedback');
    }
  }, [suggestion]);

  const applyToInput = useCallback((): string | null => {
    if (!suggestion) return null;
    return suggestion.suggested_question;
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
    requestSuggestion,
    handleFeedback,
    pendingText,
    setPendingText,
    applyToInput,
  };
}
