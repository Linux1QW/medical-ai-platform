import React, { useEffect, useState, useCallback } from 'react';
import type { CoachTraceNode } from '../api/coach';
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
 * - Never renders raw payloads, model prompts, tool arguments,
 *   or server exception messages — only the safe `safe_message` field.
 */
export const AgentTraceDrawer: React.FC<AgentTraceDrawerProps> = ({
  isAdmin,
  consultationId,
  isOpen,
  onClose,
}) => {
  const [traces, setTraces] = useState<CoachTraceNode[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchTrace = useCallback(async (cId: number) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getCoachTrace(cId);
      setTraces(data);
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

      {loading && (
        <div style={{ color: '#9ca3af', fontSize: 13, padding: '8px 0' }}>加载中...</div>
      )}

      {error && (
        <div style={{ color: '#fca5a5', fontSize: 13, padding: '8px 0' }}>{error}</div>
      )}

      {!loading && !error && traces.length === 0 && (
        <div style={{ color: '#9ca3af', fontSize: 13, padding: '8px 0' }}>暂无追踪数据</div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {traces.map((trace, idx) => (
          <div
            key={`${trace.node}-${idx}`}
            style={{
              padding: '8px 12px',
              borderRadius: 4,
              backgroundColor:
                trace.status === 'error'
                  ? '#7f1d1d'
                  : trace.status === 'completed'
                    ? '#064e3b'
                    : '#1e3a5f',
              fontSize: 13,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontWeight: 500 }}>{trace.node}</span>
              <span style={{ fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}>
                {trace.status === 'completed' ? '✓' : trace.status === 'error' ? '✗' : '…'}
                {trace.duration_ms != null && (
                  <span style={{ color: '#9ca3af' }}>{trace.duration_ms}ms</span>
                )}
              </span>
            </div>
            {/* Only render the sanitised safe_message — never raw payloads */}
            {trace.safe_message && (
              <div style={{ fontSize: 11, color: '#d1d5db', marginTop: 4 }}>
                {trace.safe_message}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
