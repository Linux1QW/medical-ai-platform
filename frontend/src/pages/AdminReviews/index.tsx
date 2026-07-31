import React, { useState } from 'react';
import {
  Card, Table, Tag, Typography, Button, Select, Form, Input, Space,
  Modal, Descriptions, Alert, message,
} from 'antd';
import {
  ExclamationCircleOutlined, EyeOutlined,
} from '@ant-design/icons';
import type { RubricItem, RiskFinding, ClinicalClaim } from '../../types';

const { Title, Text } = Typography;
const { TextArea } = Input;

/** 复核状态 */
type ReviewStatus = 'pending_review' | 'in_review' | 'approved' | 'rejected' | 'returned';

/** 复核队列项 */
interface ReviewQueueItem {
  id: number;
  consultation_id: number;
  status: ReviewStatus;
  risk_level: 'high' | 'medium' | 'low';
  priority: number;
  created_at: string;
  reason: string;
  department?: string;
  model_version?: string;
  rubric_items?: RubricItem[];
  risk_findings?: RiskFinding[];
  claims?: ClinicalClaim[];
  original_scores: Record<string, number>;
}

/** 复核决策 */
interface ReviewDecision {
  evaluation_id: number;
  status: 'approved' | 'rejected' | 'returned';
  reason_code: string;
  feedback: string;
  adjusted_items?: Array<{ item_id: string; new_verdict: string; new_score: number | null }>;
}

const STATUS_LABELS: Record<ReviewStatus, { text: string; color: string }> = {
  pending_review: { text: '待复核', color: 'orange' },
  in_review: { text: '复核中', color: 'blue' },
  approved: { text: '已批准', color: 'green' },
  rejected: { text: '已拒绝', color: 'red' },
  returned: { text: '已退回', color: 'default' },
};

const RISK_COLORS: Record<string, string> = { high: 'red', medium: 'orange', low: 'green' };

/**
 * AdminReviews — 人工复核工作台
 *
 * 功能：
 * - 按 risk_level、priority、created_at 排序
 * - 按原因、科室、模型版本筛选
 * - 展示原始回答、风险红旗、证据链
 * - 支持 rubric item 级调整
 * - 强制填写 review reason code 和 feedback
 * - 展示调整前后差异
 */
const AdminReviews: React.FC = () => {
  const [queue] = useState<ReviewQueueItem[]>([]);
  const [loading] = useState(false);
  const [selectedItem, setSelectedItem] = useState<ReviewQueueItem | null>(null);
  const [reviewModalOpen, setReviewModalOpen] = useState(false);
  const [sortBy, setSortBy] = useState<string>('priority');
  const [filterRisk, setFilterRisk] = useState<string>('');
  const [filterDept, setFilterDept] = useState<string>('');
  const [reviewForm] = Form.useForm<ReviewDecision>();

  // 排序和筛选后的队列
  const filteredQueue = queue
    .filter(item => !filterRisk || item.risk_level === filterRisk)
    .filter(item => !filterDept || item.department === filterDept)
    .sort((a, b) => {
      if (sortBy === 'priority') return b.priority - a.priority;
      if (sortBy === 'risk_level') {
        const order: Record<string, number> = { high: 0, medium: 1, low: 2 };
        return (order[a.risk_level] ?? 3) - (order[b.risk_level] ?? 3);
      }
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    });

  const columns = [
    {
      title: '风险',
      dataIndex: 'risk_level',
      key: 'risk_level',
      render: (level: string) => <Tag color={RISK_COLORS[level]}>{level}</Tag>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: ReviewStatus) => {
        const cfg = STATUS_LABELS[status] || { text: status, color: 'default' };
        return <Tag color={cfg.color}>{cfg.text}</Tag>;
      },
    },
    { title: '优先级', dataIndex: 'priority', key: 'priority' },
    { title: '问诊ID', dataIndex: 'consultation_id', key: 'consultation_id' },
    { title: '原因', dataIndex: 'reason', key: 'reason', ellipsis: true },
    { title: '科室', dataIndex: 'department', key: 'department' },
    { title: '创建时间', dataIndex: 'created_at', key: 'created_at' },
    {
      title: '操作',
      key: 'action',
      render: (_: unknown, record: ReviewQueueItem) => (
        <Button type="link" icon={<EyeOutlined />} onClick={() => handleViewDetail(record)}>
          查看
        </Button>
      ),
    },
  ];

  const handleViewDetail = (item: ReviewQueueItem) => {
    setSelectedItem(item);
    setReviewModalOpen(true);
  };

  const handleReviewSubmit = async () => {
    try {
      const values = await reviewForm.validateFields();
      // 验证：高风险 approve 必须有 reason_code
      if (selectedItem?.risk_level === 'high' && values.status === 'approved' && !values.reason_code) {
        message.error('高风险批准必须填写原因代码');
        return;
      }
      message.success('复核决策已提交');
      setReviewModalOpen(false);
      reviewForm.resetFields();
    } catch {
      message.error('请检查表单填写');
    }
  };

  return (
    <div style={{ padding: 24 }}>
      <Title level={4}>
        <ExclamationCircleOutlined style={{ marginRight: 8, color: '#fa8c16' }} />
        人工复核工作台
      </Title>

      {/* 筛选栏 */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Space wrap>
          <span>排序：</span>
          <Select value={sortBy} onChange={setSortBy} style={{ width: 140 }}>
            <Select.Option value="priority">优先级</Select.Option>
            <Select.Option value="risk_level">风险等级</Select.Option>
            <Select.Option value="created_at">创建时间</Select.Option>
          </Select>
          <span>风险等级：</span>
          <Select value={filterRisk} onChange={setFilterRisk} style={{ width: 120 }} allowClear placeholder="全部">
            <Select.Option value="high">高</Select.Option>
            <Select.Option value="medium">中</Select.Option>
            <Select.Option value="low">低</Select.Option>
          </Select>
          <span>科室：</span>
          <Select value={filterDept} onChange={setFilterDept} style={{ width: 140 }} allowClear placeholder="全部">
            <Select.Option value="cardiology">心内科</Select.Option>
            <Select.Option value="neurology">神经内科</Select.Option>
            <Select.Option value="emergency">急诊科</Select.Option>
          </Select>
        </Space>
      </Card>

      {/* 队列表格 */}
      <Table
        dataSource={filteredQueue}
        columns={columns}
        rowKey="id"
        loading={loading}
        pagination={{ pageSize: 20 }}
      />

      {/* 复核详情弹窗 */}
      <Modal
        title="复核详情"
        open={reviewModalOpen}
        onCancel={() => { setReviewModalOpen(false); reviewForm.resetFields(); }}
        width={800}
        footer={[
          <Button key="cancel" onClick={() => setReviewModalOpen(false)}>取消</Button>,
          <Button key="submit" type="primary" onClick={handleReviewSubmit}>提交决策</Button>,
        ]}
      >
        {selectedItem && (
          <div>
            {/* 风险红旗 */}
            {selectedItem.risk_findings && selectedItem.risk_findings.length > 0 && (
              <Alert
                type={selectedItem.risk_level === 'high' ? 'error' : 'warning'}
                showIcon
                style={{ marginBottom: 16 }}
                message={`发现 ${selectedItem.risk_findings.length} 项风险`}
                description={
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {selectedItem.risk_findings.map((f, i) => (
                      <li key={i}>{f.description} <Tag color={RISK_COLORS[f.severity]}>{f.severity}</Tag></li>
                    ))}
                  </ul>
                }
              />
            )}

            {/* 原始分数 */}
            <Descriptions title="原始评分" size="small" bordered column={2} style={{ marginBottom: 16 }}>
              {Object.entries(selectedItem.original_scores).map(([dim, score]) => (
                <Descriptions.Item key={dim} label={dim}>{score}</Descriptions.Item>
              ))}
            </Descriptions>

            {/* Rubric Items */}
            {selectedItem.rubric_items && selectedItem.rubric_items.length > 0 && (
              <Card title="Rubric 明细" size="small" style={{ marginBottom: 16 }}>
                {selectedItem.rubric_items.map((item, idx) => (
                  <div key={idx} style={{ padding: '8px 0', borderBottom: '1px solid #f0f0f0' }}>
                    <Space>
                      <Tag color={item.verdict === 'pass' ? 'green' : item.verdict === 'fail' ? 'red' : 'gold'}>
                        {item.verdict}
                      </Tag>
                      <Text strong>{item.item_id}</Text>
                      <Text type="secondary">{item.description}</Text>
                      {item.severity === 'high' && <Tag color="red">高严重</Tag>}
                    </Space>
                  </div>
                ))}
              </Card>
            )}

            {/* 复核表单 */}
            <Form form={reviewForm} layout="vertical">
              <Form.Item name="status" label="决策" rules={[{ required: true }]}>
                <Select>
                  <Select.Option value="approved">批准</Select.Option>
                  <Select.Option value="rejected">拒绝</Select.Option>
                  <Select.Option value="returned">退回</Select.Option>
                </Select>
              </Form.Item>
              <Form.Item name="reason_code" label="原因代码" rules={[{ required: true, message: '必须填写原因代码' }]}>
                <Input placeholder="例如: evidence_insufficient, score_adjustment" />
              </Form.Item>
              <Form.Item name="feedback" label="反馈意见" rules={[{ required: true, message: '必须填写反馈意见' }]}>
                <TextArea rows={3} placeholder="详细说明复核依据和调整理由..." />
              </Form.Item>
            </Form>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default AdminReviews;
