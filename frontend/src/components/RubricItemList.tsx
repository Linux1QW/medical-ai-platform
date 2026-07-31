import React from 'react';
import { Tag, Collapse, Typography, Empty } from 'antd';
import { CheckCircleOutlined, MinusCircleOutlined, CloseCircleOutlined, QuestionCircleOutlined, StopOutlined } from '@ant-design/icons';
import type { RubricItem, RubricVerdict } from '../types';

const { Text, Paragraph } = Typography;

const VERDICT_CONFIG: Record<RubricVerdict, { label: string; color: string; icon: React.ReactNode }> = {
  pass: { label: '通过', color: 'green', icon: <CheckCircleOutlined /> },
  partial: { label: '部分通过', color: 'gold', icon: <MinusCircleOutlined /> },
  fail: { label: '未通过', color: 'red', icon: <CloseCircleOutlined /> },
  not_applicable: { label: '不适用', color: 'default', icon: <StopOutlined /> },
  unassessed: { label: '未评估', color: 'default', icon: <QuestionCircleOutlined /> },
};

interface RubricItemListProps {
  items: RubricItem[];
  dimension?: string;
}

/**
 * RubricItemList — 展示原子 Rubric 项列表
 * 
 * 关键语义：
 * - unassessed 显示"未评估"，不显示 0 分
 * - high severity + fail 自动标记 review_required
 */
export const RubricItemList: React.FC<RubricItemListProps> = ({ items, dimension }) => {
  const filtered = dimension ? items.filter(i => i.dimension === dimension) : items;

  if (filtered.length === 0) {
    return <Empty description="暂无 Rubric 项" image={Empty.PRESENTED_IMAGE_SIMPLE} />;
  }

  return (
    <Collapse
      size="small"
      items={filtered.map(item => {
        const cfg = VERDICT_CONFIG[item.verdict] || VERDICT_CONFIG.unassessed;
        const isHighSeverityFail = item.severity === 'high' && item.verdict === 'fail';
        
        return {
          key: item.item_id,
          label: (
            <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ color: cfg.color === 'green' ? '#52c41a' : cfg.color === 'red' ? '#ff4d4f' : cfg.color === 'gold' ? '#faad14' : '#999' }}>
                {cfg.icon}
              </span>
              <Text strong style={{ fontSize: 13 }}>{item.item_id}</Text>
              <Tag color={cfg.color} style={{ borderRadius: 10 }}>{cfg.label}</Tag>
              {item.severity === 'high' && <Tag color="red">高严重</Tag>}
              {isHighSeverityFail && <Tag color="orange">需复核</Tag>}
              {/* 关键：unassessed 不渲染分数为 0 */}
              {item.verdict !== 'unassessed' && item.score !== null && (
                <Text type="secondary" style={{ fontSize: 12 }}>{item.score}分</Text>
              )}
              {item.verdict === 'unassessed' && (
                <Text type="secondary" style={{ fontSize: 12, fontStyle: 'italic' }}>未评估</Text>
              )}
            </span>
          ),
          children: (
            <div>
              <Paragraph style={{ margin: '0 0 8px', lineHeight: 1.6 }}>{item.description}</Paragraph>
              {item.evidence_spans.length > 0 && (
                <div style={{ marginBottom: 8 }}>
                  <Text type="secondary" style={{ fontSize: 12 }}>证据片段：</Text>
                  {item.evidence_spans.map((span: string, idx: number) => (
                    <div key={idx} style={{ padding: '4px 8px', background: '#fafafa', borderRadius: 4, marginTop: 4, fontSize: 12 }}>
                      {span}
                    </div>
                  ))}
                </div>
              )}
              {item.citation_ids.length > 0 && (
                <div>
                  <Text type="secondary" style={{ fontSize: 12 }}>引用：</Text>
                  {item.citation_ids.map((id: string) => (
                    <Tag key={id} style={{ fontSize: 11 }}>{id}</Tag>
                  ))}
                </div>
              )}
              {item.citation_ids.length === 0 && item.verdict !== 'not_applicable' && (
                <Text type="secondary" style={{ fontSize: 12, fontStyle: 'italic' }}>无可验证证据</Text>
              )}
            </div>
          ),
        };
      })}
    />
  );
};

export default RubricItemList;
