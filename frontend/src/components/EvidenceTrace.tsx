import React from 'react';
import { Tag, Typography, Collapse, Empty } from 'antd';
import { LinkOutlined } from '@ant-design/icons';
import type { ClinicalClaim, EvidenceLink, ClaimStatus } from '../types';

const { Text, Paragraph } = Typography;

const STATUS_CONFIG: Record<ClaimStatus, { label: string; color: string }> = {
  supported: { label: '有证据支持', color: 'green' },
  partially_supported: { label: '部分支持', color: 'gold' },
  unsupported: { label: '无证据支持', color: 'red' },
  conflicting: { label: '证据冲突', color: 'magenta' },
};

const LINK_TYPE_LABELS: Record<string, { label: string; color: string }> = {
  supports: { label: '支持', color: 'green' },
  contradicts: { label: '矛盾', color: 'red' },
  insufficient: { label: '不足', color: 'default' },
};

interface EvidenceTraceProps {
  claims: ClinicalClaim[];
}

/**
 * EvidenceTrace — 展示临床主张和证据链
 * 
 * 关键语义：
 * - unsupported treatment/risk claim 显示风险标记
 * - 冲突证据不能标记为 supported
 * - 无引用时显示"无可验证证据"
 */
export const EvidenceTrace: React.FC<EvidenceTraceProps> = ({ claims }) => {
  if (claims.length === 0) {
    return <Empty description="暂无临床主张" image={Empty.PRESENTED_IMAGE_SIMPLE} />;
  }

  return (
    <Collapse
      size="small"
      items={claims.map(claim => {
        const statusCfg = STATUS_CONFIG[claim.status] || STATUS_CONFIG.unsupported;
        const isUnsupportedRisk = (claim.claim_type === 'treatment' || claim.claim_type === 'risk') && claim.status === 'unsupported';
        const hasConflictingEvidence = claim.evidence_links.some(l => l.link_type === 'contradicts');

        return {
          key: claim.claim_id,
          label: (
            <span style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <Tag color={statusCfg.color}>{statusCfg.label}</Tag>
              <Tag>{claim.claim_type}</Tag>
              <Text style={{ fontSize: 13, flex: 1 }}>{claim.content}</Text>
              {isUnsupportedRisk && <Tag color="red">需复核</Tag>}
              {hasConflictingEvidence && <Tag color="magenta">证据冲突</Tag>}
            </span>
          ),
          children: (
            <div>
              {claim.evidence_links.length > 0 ? (
                claim.evidence_links.map((link: EvidenceLink, idx: number) => {
                  const linkCfg = LINK_TYPE_LABELS[link.link_type] || LINK_TYPE_LABELS.insufficient;
                  return (
                    <div key={idx} style={{
                      padding: '8px 12px',
                      borderLeft: `3px solid ${link.link_type === 'supports' ? '#52c41a' : link.link_type === 'contradicts' ? '#ff4d4f' : '#d9d9d9'}`,
                      background: '#fafafa',
                      borderRadius: 4,
                      marginBottom: 8,
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                        <LinkOutlined style={{ color: '#1677ff' }} />
                        <Tag color={linkCfg.color}>{linkCfg.label}</Tag>
                        <Text type="secondary" style={{ fontSize: 11 }}>
                          置信度: {(link.entailment_score * 100).toFixed(0)}%
                        </Text>
                        <Text type="secondary" style={{ fontSize: 11 }}>
                          Citation: {link.citation_id}
                        </Text>
                      </div>
                      <Paragraph style={{ fontSize: 12, margin: 0, color: '#555' }} ellipsis={{ rows: 2 }}>
                        {link.evidence_span}
                      </Paragraph>
                    </div>
                  );
                })
              ) : (
                <Text type="secondary" style={{ fontStyle: 'italic' }}>无可验证证据</Text>
              )}
            </div>
          ),
        };
      })}
    />
  );
};

export default EvidenceTrace;
