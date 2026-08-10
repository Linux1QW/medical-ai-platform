import React, { useState, useCallback } from 'react';

interface VoiceSession {
  session_id: string;
  room_name: string;
  room_token: string;
  status: string;
  expires_at: number;
}

interface VoiceConsultationProps {
  consultationId: number;
  doctorId: number;
}

export const VoiceConsultation: React.FC<VoiceConsultationProps> = ({
  consultationId,
  doctorId,
}) => {
  const [session, setSession] = useState<VoiceSession | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);

  const startSession = useCallback(async () => {
    setIsConnecting(true);
    setError(null);
    try {
      const res = await fetch('/api/v1/voice/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ consultation_id: consultationId, doctor_id: doctorId }),
      });
      if (!res.ok) throw new Error(`Failed to create voice session: ${res.status}`);
      const data = await res.json();
      setSession(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsConnecting(false);
    }
  }, [consultationId, doctorId]);

  const endSession = useCallback(async () => {
    if (!session) return;
    try {
      await fetch(`/api/v1/voice/sessions/${session.room_name}/end`, { method: 'POST' });
      setSession(null);
    } catch (err: any) {
      setError(err.message);
    }
  }, [session]);

  return (
    <div className="voice-consultation" role="region" aria-label="语音问诊">
      <h3>语音问诊 (Beta)</h3>

      {!session && (
        <button
          onClick={startSession}
          disabled={isConnecting}
          style={{ padding: '8px 16px', backgroundColor: '#3b82f6', color: 'white', borderRadius: 4 }}
        >
          {isConnecting ? '连接中...' : '开始语音问诊'}
        </button>
      )}

      {session && (
        <div>
          <p>房间: {session.room_name}</p>
          <p>状态: {session.status}</p>
          <p style={{ fontSize: 12, color: '#6b7280' }}>
            Token 有效期: 10 分钟
          </p>
          <button
            onClick={endSession}
            style={{ padding: '8px 16px', backgroundColor: '#ef4444', color: 'white', borderRadius: 4 }}
          >
            结束语音
          </button>
        </div>
      )}

      {error && <p style={{ color: '#ef4444' }}>{error}</p>}

      <p style={{ fontSize: 11, color: '#9ca3af', marginTop: 8 }}>
        注意：语音功能不会保存原始音频数据。
      </p>
    </div>
  );
};
