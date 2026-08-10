import React from 'react';

interface TraceNode {
  node: string;
  status: 'started' | 'completed' | 'error';
  error?: string;
}

interface AgentTraceDrawerProps {
  traces: TraceNode[];
  isOpen: boolean;
  onClose: () => void;
}

export const AgentTraceDrawer: React.FC<AgentTraceDrawerProps> = ({ traces, isOpen, onClose }) => {
  if (!isOpen) return null;

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
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <h3 style={{ margin: 0 }}>Agent Trace</h3>
        <button
          onClick={onClose}
          aria-label="关闭追踪面板"
          style={{ background: 'none', color: '#f9fafb', border: '1px solid #4b5563', borderRadius: 4, padding: '4px 8px' }}
        >
          ✕
        </button>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {traces.map((trace, idx) => (
          <div
            key={idx}
            style={{
              padding: '8px 12px',
              borderRadius: 4,
              backgroundColor:
                trace.status === 'error' ? '#7f1d1d' : trace.status === 'completed' ? '#064e3b' : '#1e3a5f',
              fontSize: 13,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontWeight: 500 }}>{trace.node}</span>
              <span style={{ fontSize: 11 }}>
                {trace.status === 'completed' ? '✓' : trace.status === 'error' ? '✗' : '…'}
              </span>
            </div>
            {trace.error && (
              <div style={{ fontSize: 11, color: '#fca5a5', marginTop: 4 }}>{trace.error}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
