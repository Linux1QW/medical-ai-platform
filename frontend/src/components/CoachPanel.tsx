import React, { useState } from 'react';
import type { CoachPanelStatus } from '../hooks/useCoachSuggestion';
import { useCoachSuggestion } from '../hooks/useCoachSuggestion';

interface CoachPanelProps {
  consultationId: number;
  /** Called when user clicks "填入输入框". Parent sets the composer text. NEVER auto-sends. */
  onApplySuggestion: (text: string) => void;
  /** Disable interactions (e.g. when consultation has ended or a message is being sent). */
  disabled?: boolean;
}

const STATUS_LABELS: Record<CoachPanelStatus, string> = {
  disabled: '教练已关闭',
  idle: '等待问诊',
  thinking: '正在分析...',
  suggestion: '建议就绪',
  degraded: '服务降级',
  error: '连接错误',
};

const STATUS_COLORS: Record<CoachPanelStatus, string> = {
  disabled: '#9ca3af',
  idle: '#6b7280',
  thinking: '#3b82f6',
  suggestion: '#10b981',
  degraded: '#f59e0b',
  error: '#ef4444',
};

export const CoachPanel: React.FC<CoachPanelProps> = ({
  consultationId,
  onApplySuggestion,
  disabled = false,
}) => {
  const {
    status,
    suggestion,
    errorMessage,
    turnNo,
    requestSuggestion,
    retryLastSuggestion,
    handleFeedback,
    dismissSuggestion,
  } = useCoachSuggestion(consultationId);

  // Local composer text for the "hint request" textarea
  const [pendingText, setPendingText] = useState('');

  const handleHintRequest = () => {
    if (pendingText.trim() && !disabled) {
      requestSuggestion(pendingText.trim());
    }
  };

  const handleApply = () => {
    if (!suggestion) return;
    // Pass the suggested text to the parent composer — NEVER calls handleSend
    onApplySuggestion(suggestion.suggested_question);
  };

  return (
    <div
      className="coach-panel"
      role="complementary"
      aria-label="问诊教练面板"
      style={{ border: `2px solid ${STATUS_COLORS[status]}`, borderRadius: 8, padding: 16 }}
    >
      {/* Status header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <span
          style={{
            width: 10,
            height: 10,
            borderRadius: '50%',
            backgroundColor: STATUS_COLORS[status],
            display: 'inline-block',
          }}
          aria-hidden="true"
        />
        <strong aria-live="polite">{STATUS_LABELS[status]}</strong>
        {turnNo > 0 && <span style={{ fontSize: 12, color: '#6b7280' }}>Turn {turnNo}</span>}
      </div>

      {/* Disabled state */}
      {status === 'disabled' && (
        <p style={{ color: '#9ca3af' }}>教练功能已关闭。请联系管理员启用。</p>
      )}

      {/* Idle — user can type and request a hint */}
      {status === 'idle' && (
        <div>
          <textarea
            value={pendingText}
            onChange={(e) => setPendingText(e.target.value)}
            placeholder='输入你想问患者的问题，点击"给我一个提示"'
            rows={3}
            disabled={disabled}
            style={{ width: '100%', marginBottom: 8, resize: 'vertical' }}
            aria-label="教练输入框"
          />
          <button
            onClick={handleHintRequest}
            disabled={!pendingText.trim() || disabled}
            style={{
              backgroundColor: '#3b82f6',
              color: 'white',
              padding: '6px 16px',
              borderRadius: 4,
              cursor: disabled ? 'not-allowed' : 'pointer',
            }}
          >
            给我一个提示
          </button>
        </div>
      )}

      {/* Thinking */}
      {status === 'thinking' && (
        <div aria-live="polite" style={{ color: '#3b82f6' }}>
          <span className="thinking-indicator">正在分析您的问诊内容...</span>
        </div>
      )}

      {/* Suggestion ready */}
      {status === 'suggestion' && suggestion && (
        <div aria-live="polite">
          <div style={{ backgroundColor: '#f0fdf4', padding: 12, borderRadius: 4, marginBottom: 8 }}>
            <p style={{ fontWeight: 600, marginBottom: 4 }}>建议问题：</p>
            <p>{suggestion.suggested_question}</p>
            <p style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>
              {suggestion.rationale_summary}
            </p>
            <div style={{ marginTop: 4, fontSize: 12 }}>
              <span>置信度: {(suggestion.confidence * 100).toFixed(0)}%</span>
              <span style={{ marginLeft: 8 }}>风险: {suggestion.risk_level}</span>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button
              onClick={handleApply}
              disabled={disabled}
              style={{
                backgroundColor: '#10b981',
                color: 'white',
                padding: '4px 12px',
                borderRadius: 4,
              }}
            >
              填入输入框
            </button>
            <button
              onClick={() => handleFeedback({ feedback: 'accepted' })}
              style={{ padding: '4px 12px', borderRadius: 4 }}
              title="采纳该建议"
            >
              👍 有用
            </button>
            <button
              onClick={() => handleFeedback({ feedback: 'rejected' })}
              style={{ padding: '4px 12px', borderRadius: 4 }}
              title="不采纳该建议"
            >
              👎 没用
            </button>
            <button
              onClick={dismissSuggestion}
              style={{ padding: '4px 12px', borderRadius: 4 }}
              title="忽略此建议，返回等待状态"
            >
              ✕ 忽略
            </button>
          </div>
        </div>
      )}

      {/* Degraded */}
      {status === 'degraded' && (
        <div style={{ color: '#f59e0b' }}>
          <p>教练服务暂时降级。</p>
          {errorMessage && <p style={{ fontSize: 12 }}>{errorMessage}</p>}
          <button
            onClick={retryLastSuggestion}
            disabled={disabled}
            style={{ marginTop: 8 }}
          >
            重试
          </button>
        </div>
      )}

      {/* Error */}
      {status === 'error' && (
        <div style={{ color: '#ef4444' }}>
          <p>连接错误，请刷新页面。</p>
          {errorMessage && <p style={{ fontSize: 12 }}>{errorMessage}</p>}
        </div>
      )}
    </div>
  );
};
