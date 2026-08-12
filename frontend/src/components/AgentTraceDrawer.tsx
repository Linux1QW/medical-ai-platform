import React, { useEffect, useState, useCallback } from 'react';
import type { CoachTraceResponse } from '../api/coach';
import { getCoachTrace } from '../api/coach';

interface AgentTraceDrawerProps {
  /** Only render for authorized admins. */
  isAdmin: boolean;
  consultationId: number | null;
  isOpen: boolean;
  onClose: () => void;
}

/**
 * Sanitised agent-trace drawer.
 *
 * - Mounts only for authorized admins (`isAdmin` guard).
 * - Fetches the sanitised trace projection from the backend.
 * - Renders session info, decisions, and events from the CoachTraceResponse.
 * - Never renders raw payloads, model prompts, tool arguments,
 *   or server exception messages.
 */
export const AgentTraceDrawer: React.FC<AgentTraceDrawerProps> = ({
  isAdmin,
  consultationId,
  isOpen,
  onClose,
}) => {
  const [trace, setTrace] = useState<CoachTraceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchTrace = useCallback(async (cId: number) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getCoachTrace(cId);
      setTrace(data);
    } catch {
      setError('加载追踪数据失败');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!isOpen || !isAdmin || consultationId == null) return;
    fetchTrace(consultationId);
  }, [isOpen, isAdmin, consultationId, fetchTrace]);

  // Non-admin guard: never render
  if (!isAdmin || !isOpen) return null;

  const decisions = trace?.decisions ?? [];
  const events = trace?.events ?? [];
  const hasData = decisions.length > 0 || events.length > 0;

  return (
    <div
      className="agent-trace-drawer"
      role="dialog"
      aria-label="Agent 执行追踪"
      style={{
        position: 'fixed',
        right: 0,
        top: 0,
        bottom: 0,
        width: 360,
        backgroundColor: '#1f2937',
        color: '#f9fafb',
        padding: 16,
        overflowY: 'auto',
        zIndex: 1000,
        boxShadow: '-4px 0 16px rgba(0,0,0,0.3)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h3 style={{ margin: 0, fontSize: 16 }}>Agent Trace</h3>
        <button
          onClick={onClose}
          aria-label="关闭追踪面板"
          style={{
            background: 'none',
            color: '#f9fafb',
            border: '1px solid #4b5563',
            borderRadius: 4,
            padding: '4px 8px',
            cursor: 'pointer',
          }}
        >
          ✕
        </button>
      </div>

      {trace?.session && (
        <div style={{ padding: '8px 12px', backgroundColor: '#374151', borderRadius: 4, marginBottom: 8, fontSize: 13 }}>
          <div style={{ fontWeight: 500 }}>Session: {trace.session.session_id}</div>
          <div style={{ fontSize: 11, color: '#9ca3af' }}>Status: {trace.session.status}</div>
        </div>
      )}

      {loading && (
        <div style={{ color: '#9ca3af', fontSize: 13, padding: '8px 0' }}>加载中...</div>
      )}

      {error && (
        <div style={{ color: '#fca5a5', fontSize: 13, padding: '8px 0' }}>{error}</div>
      )}

      {!loading && !error && !hasData && (
        <div style={{ color: '#9ca3af', fontSize: 13, padding: '8px 0' }}>暂无追踪数据</div>
      )}

      {/* Decisions */}
      {decisions.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <h4 style={{ fontSize: 13, color: '#9ca3af', margin: '8px 0 4px' }}>Decisions</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {decisions.map((d) => (
              <div
                key={d.decision_id}
                style={{
                  padding: '8px 12px',
                  borderRadius: 4,
                  backgroundColor: '#1e3a5f',
                  fontSize: 13,
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 500 }}>{d.agent}</span>
                  {d.timestamp && (
                    <span style={{ fontSize: 11, color: '#9ca3af' }}>{d.timestamp}</span>
                  )}
                </div>
                <div style={{ fontSize: 12, color: '#d1d5db', marginTop: 4 }}>{d.action}</div>
                {d.rationale && (
                  <div style={{ fontSize: 11, color: '#9ca3af', marginTop: 4 }}>{d.rationale}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Events */}
      {events.length > 0 && (
        <div>
          <h4 style={{ fontSize: 13, color: '#9ca3af', margin: '8px 0 4px' }}>Events</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {events.map((e) => (
              <div
                key={e.event_id}
                style={{
                  padding: '8px 12px',
                  borderRadius: 4,
                  backgroundColor:
                    e.event_type === 'error'
                      ? '#7f1d1d'
                      : e.event_type === 'completed'
                        ? '#064e3b'
                        : '#1e3a5f',
                  fontSize: 13,
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 500 }}>{e.agent}</span>
                  <span style={{ fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}>
                    {e.event_type}
                    {e.timestamp && (
                      <span style={{ color: '#9ca3af' }}>{e.timestamp}</span>
                    )}
                  </span>
                </div>
                {e.message && (
                  <div style={{ fontSize: 11, color: '#d1d5db', marginTop: 4 }}>
                    {e.message}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
