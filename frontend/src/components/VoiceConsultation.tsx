import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Room, RoomEvent } from 'livekit-client';
import request from '../utils/request';

interface VoiceSession {
  session_id: string;
  room_name: string;
  room_token: string;
  consultation_id: number;
  status: string;
  expires_at: number;
}

type ConnectionState = 'idle' | 'connecting' | 'connected' | 'disconnected' | 'error';

interface VoiceConsultationProps {
  consultationId: number;
  doctorId?: number;
}

/** LiveKit server URL — read from Vite env or fallback to localhost. */
const LIVEKIT_URL = import.meta.env.VITE_LIVEKIT_URL?.trim() || 'ws://localhost:7880';

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : '未知语音服务错误';
}

export const VoiceConsultation: React.FC<VoiceConsultationProps> = ({
  consultationId,
}) => {
  const [session, setSession] = useState<VoiceSession | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>('idle');
  const [transcriptState, setTranscriptState] = useState<string>('等待连接...');
  const roomRef = useRef<Room | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (roomRef.current) {
        roomRef.current.disconnect();
        roomRef.current = null;
      }
    };
  }, []);

  const startSession = useCallback(async () => {
    setConnectionState('connecting');
    setError(null);
    setTranscriptState('正在创建会话...');
    try {
      // Use unified request client (reads sessionStorage token automatically)
      const data = await request.post<{
        session_id: string;
        room_name: string;
        room_token: string;
        consultation_id: number;
        status: string;
        expires_at: number;
      }>('/voice/sessions', { consultation_id: consultationId });

      setSession(data as unknown as VoiceSession);

      // Connect to LiveKit Room
      const room = new Room();

      // Listen for room events
      room.on(RoomEvent.Disconnected, (reason) => {
        setConnectionState('disconnected');
        setTranscriptState(reason ? `已断开: ${reason}` : '已断开');
        roomRef.current = null;
      });

      room.on(RoomEvent.Reconnecting, () => {
        setTranscriptState('正在重新连接...');
      });

      room.on(RoomEvent.Reconnected, () => {
        setConnectionState('connected');
        setTranscriptState('已重新连接');
      });

      // Attempt real connection
      await room.connect(LIVEKIT_URL, (data as unknown as VoiceSession).room_token);
      roomRef.current = room;
      setConnectionState('connected');
      setTranscriptState('已连接，等待语音输入...');
    } catch (err: unknown) {
      setConnectionState('error');
      setError(getErrorMessage(err));
      // Clean up room on failure
      if (roomRef.current) {
        roomRef.current.disconnect();
        roomRef.current = null;
      }
    }
  }, [consultationId]);

  const endSession = useCallback(async () => {
    if (!session) return;
    try {
      await request.post(`/voice/sessions/${session.room_name}/end`);

      // Disconnect LiveKit room
      if (roomRef.current) {
        roomRef.current.disconnect();
        roomRef.current = null;
      }
      setSession(null);
      setConnectionState('idle');
      setTranscriptState('等待连接...');
    } catch (err: unknown) {
      // Failed end must NOT falsely clear local session state
      setError(getErrorMessage(err));
    }
  }, [session]);

  const reconnect = useCallback(async () => {
    if (!session) {
      await startSession();
      return;
    }
    // Disconnect existing room if any
    if (roomRef.current) {
      roomRef.current.disconnect();
      roomRef.current = null;
    }
    setConnectionState('connecting');
    setError(null);
    setTranscriptState('正在重新连接...');
    try {
      const room = new Room();

      room.on(RoomEvent.Disconnected, (reason) => {
        setConnectionState('disconnected');
        setTranscriptState(reason ? `已断开: ${reason}` : '已断开');
        roomRef.current = null;
      });

      room.on(RoomEvent.Reconnecting, () => {
        setTranscriptState('正在重新连接...');
      });

      room.on(RoomEvent.Reconnected, () => {
        setConnectionState('connected');
        setTranscriptState('已重新连接');
      });

      await room.connect(LIVEKIT_URL, session.room_token);
      roomRef.current = room;
      setConnectionState('connected');
      setTranscriptState('已重新连接');
    } catch (err: unknown) {
      setConnectionState('error');
      setError(getErrorMessage(err));
      if (roomRef.current) {
        roomRef.current.disconnect();
        roomRef.current = null;
      }
    }
  }, [session, startSession]);

  const statusLabel: Record<ConnectionState, string> = {
    idle: '未连接',
    connecting: '连接中...',
    connected: '已连接',
    disconnected: '已断开',
    error: '连接错误',
  };

  return (
    <div className="voice-consultation" role="region" aria-label="语音问诊">
      <h3>语音问诊 (Beta)</h3>

      <div style={{ marginBottom: 8 }}>
        <span style={{ fontSize: 12, color: '#6b7280' }}>
          状态: {statusLabel[connectionState]}
        </span>
        {session && (
          <span style={{ fontSize: 12, color: '#6b7280', marginLeft: 12 }}>
            房间: {session.room_name}
          </span>
        )}
      </div>

      <p style={{ fontSize: 12, color: '#374151' }}>{transcriptState}</p>

      {connectionState === 'idle' && (
        <button
          onClick={startSession}
          style={{ padding: '8px 16px', backgroundColor: '#3b82f6', color: 'white', borderRadius: 4 }}
        >
          开始语音问诊
        </button>
      )}

      {connectionState === 'error' && (
        <button
          onClick={reconnect}
          style={{ padding: '8px 16px', backgroundColor: '#f59e0b', color: 'white', borderRadius: 4, marginRight: 8 }}
        >
          重新连接
        </button>
      )}

      {(connectionState === 'connected' || connectionState === 'error') && session && (
        <button
          onClick={endSession}
          style={{ padding: '8px 16px', backgroundColor: '#ef4444', color: 'white', borderRadius: 4 }}
        >
          结束语音
        </button>
      )}

      {error && (
        <p style={{ color: '#ef4444', marginTop: 8, fontSize: 13 }}>{error}</p>
      )}

      <p style={{ fontSize: 11, color: '#9ca3af', marginTop: 8 }}>
        注意：语音功能不会保存原始音频数据。Token 有效期: 10 分钟
      </p>
    </div>
  );
};
