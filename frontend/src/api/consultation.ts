import request from '../utils/request';
import type { Consultation, ConsultationDetail, Message } from '../types';

export const startConsultation = (patient_id: number): Promise<Consultation> =>
  request.post('/consultations/', { patient_id });

export const getConsultations = (): Promise<Consultation[]> =>
  request.get('/consultations/');

export interface ConsultationQueryParams {
  username?: string;
  personality?: string;
  start_time?: string;
  end_time?: string;
}

export const getAllConsultations = (params?: ConsultationQueryParams): Promise<Consultation[]> =>
  request.get('/consultations/all', { params });

export const getConsultationDetail = (id: number): Promise<ConsultationDetail> =>
  request.get(`/consultations/${id}`);

export const sendMessage = (consultation_id: number, content: string): Promise<Message[]> =>
  request.post(`/consultations/${consultation_id}/messages`, { content });

// SSE 事件类型
export interface SSEProgressEvent {
  step: string;
  message: string;
  progress: number;
}

export interface SSECompleteEvent {
  doctor_msg: Message;
  patient_msg: Message;
}

export interface SSEErrorEvent {
  message: string;
}

export interface SendMessageStreamCallbacks {
  onProgress?: (event: SSEProgressEvent) => void;
  onComplete?: (event: SSECompleteEvent) => void;
  onError?: (event: SSEErrorEvent) => void;
}

/**
 * SSE 流式发送消息
 * 使用 fetch API 处理 POST 请求的 SSE 响应
 */
export const sendMessageStream = async (
  consultation_id: number,
  content: string,
  callbacks: SendMessageStreamCallbacks,
  signal?: AbortSignal,
): Promise<void> => {
  const token = sessionStorage.getItem('token');
  const response = await fetch(`/api/v1/consultations/${consultation_id}/messages/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
      'Accept': 'text/event-stream',
    },
    body: JSON.stringify({ content }),
    signal,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: '请求失败' }));
    throw new Error(errorData.detail || errorData.message || `HTTP ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('无法读取响应流');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // 解析 SSE 事件（以双换行分隔）
      const events = buffer.split('\n\n');
      buffer = events.pop() || ''; // 最后一个可能是不完整的事件

      for (const eventText of events) {
        if (!eventText.trim()) continue;

        let eventType = 'message';
        let eventData = '';

        // 解析 event 和 data 行
        const lines = eventText.split('\n');
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            eventData = line.slice(6);
          }
        }

        if (!eventData) continue;

        try {
          const data = JSON.parse(eventData);
          switch (eventType) {
            case 'progress':
              callbacks.onProgress?.(data as SSEProgressEvent);
              break;
            case 'complete':
              callbacks.onComplete?.(data as SSECompleteEvent);
              break;
            case 'error':
              callbacks.onError?.(data as SSEErrorEvent);
              break;
          }
        } catch {
          console.warn('SSE 事件解析失败:', eventData);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
};

export const submitDiagnosis = (
  consultation_id: number,
  diagnosis: string,
  treatment_plan: string,
): Promise<Consultation> =>
  request.post(`/consultations/${consultation_id}/submit-diagnosis`, {
    diagnosis,
    treatment_plan,
  });

export const endConsultation = (id: number): Promise<Consultation> =>
  request.post(`/consultations/${id}/end`);

export const extendRounds = (id: number): Promise<Consultation> =>
  request.post(`/consultations/${id}/extend`);

export const deleteConsultation = (id: number): Promise<{ detail: string }> =>
  request.delete(`/consultations/${id}`);
