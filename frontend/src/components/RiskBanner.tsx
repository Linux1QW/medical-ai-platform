import React from 'react';
import { Alert, Tag, Typography, Space } from 'antd';
import { WarningOutlined, ExclamationCircleOutlined } from '@ant-design/icons';
import type { RiskFinding, RiskType } from '../types';

const { Text } = Typography;

const RISK_TYPE_LABELS: Record<RiskType, string> = {
  emergency: '急诊',
  medication: '用药',
  population: '人群',
  privacy: '隐私',
  evidence_conflict: '证据冲突',
};

const SEVERITY_CONFIG: Record<string, { type: 'error' | 'warning' | 'info'; color: string }> = {
  high: { type: 'error', color: '#ff4d4f' },
  medium: { type: 'warning', color: '#faad14' },
  low: { type: 'info', color: '#1677ff' },
};

interface RiskBannerProps {
  findings: RiskFinding[];
}

/**
 * RiskBanner — 展示风险发现横幅
 * 
 * 关键语义：
 * - high severity 显示为 error 级别 Alert
 * - 所有高危 finding 需标记 needs_review
 */
export const RiskBanner: React.FC<RiskBannerProps> = ({ findings }) => {
  if (findings.length === 0) {
    return null;
  }

  // 按严重度排序：high > medium > low
  const sorted = [...findings].sort((a: RiskFinding, b: RiskFinding) => {
    const order: Record<string, number> = { high: 0, medium: 1, low: 2 };
    return (order[a.severity] ?? 3) - (order[b.severity] ?? 3);
  });

  const hasHighRisk = sorted.some(f => f.severity === 'high');
  const alertType = hasHighRisk ? 'error' : 'warning';

  return (
    <Alert
      type={alertType}
      showIcon
      icon={hasHighRisk ? <ExclamationCircleOutlined /> : <WarningOutlined />}
      style={{ marginBottom: 16 }}
      message={
        <span>
          发现 {findings.length} 项风险
          {hasHighRisk && <Tag color="red" style={{ marginLeft: 8 }}>高危</Tag>}
        </span>
      }
      description={
        <Space direction="vertical" style={{ width: '100%' }}>
          {sorted.map(finding => {
            const cfg = SEVERITY_CONFIG[finding.severity] || SEVERITY_CONFIG.low;
            return (
              <div key={finding.finding_id} style={{ 
                padding: '8px 12px', 
                background: 'rgba(255,255,255,0.8)', 
                borderRadius: 4,
                borderLeft: `3px solid ${cfg.color}`,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                  <Tag color={cfg.color}>{finding.severity}</Tag>
                  <Tag>{RISK_TYPE_LABELS[finding.risk_type] || finding.risk_type}</Tag>
                  {finding.needs_review && <Tag color="orange">需复核</Tag>}
                </div>
                <Text style={{ fontSize: 13 }}>{finding.description}</Text>
                {finding.evidence_span && (
                  <div style={{ marginTop: 4, fontSize: 12, color: '#666' }}>
                    证据: {finding.evidence_span}
                  </div>
                )}
                {finding.policy_action && (
                  <div style={{ marginTop: 4, fontSize: 12, color: '#1677ff' }}>
                    策略动作: {finding.policy_action}
                  </div>
                )}
              </div>
            );
          })}
        </Space>
      }
    />
  );
};

export default RiskBanner;
