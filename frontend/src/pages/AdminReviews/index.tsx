import React, { useState, useEffect, useCallback } from 'react';
import {
  Card, Table, Tag, Typography, Button, Form, Input, Space,
  Modal, Descriptions, Alert, message,
} from 'antd';
import {
  ExclamationCircleOutlined, EyeOutlined, ReloadOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import type { PendingReviewItem, Evaluation, ScoreAdjustments } from '../../types';
import { listPendingReviews, submitReview } from '../../api/review';
import { getEvaluation } from '../../api/evaluation';

const { Title, Text } = Typography;
const { TextArea } = Input;

/**
 * AdminReviews — 人工复核工作台
 *
 * 功能：
 * - 展示待复核列表（从后端 API 获取）
 * - 查看评估详情（分数、引用、检索状态）
 * - 可选五维分数调整
 * - 提交复核反馈
 */
const AdminReviews: React.FC = () => {
  const [queue, setQueue] = useState<PendingReviewItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedItem, setSelectedItem] = useState<PendingReviewItem | null>(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [conflictError, setConflictError] = useState<string | null>(null);
  const [reviewForm] = Form.useForm<{ feedback: string; inquiry_score?: string; knowledge_score?: string; humanistic_score?: string; diagnosis_score?: string; treatment_score?: string }>();

  const fetchQueue = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const res = await listPendingReviews();
      setQueue(res.items);
    } catch {
      setLoadError('加载失败，请重试');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchQueue();
  }, [fetchQueue]);

  const handleViewDetail = async (item: PendingReviewItem) => {
    setSelectedItem(item);
    setDetailModalOpen(true);
    setConflictError(null);
    setDetailLoading(true);
    try {
      const evalData = await getEvaluation(item.consultation_id);
      setEvaluation(evalData);
    } catch {
      message.error('获取评估详情失败');
    } finally {
      setDetailLoading(false);
    }
  };

  const handleReviewSubmit = async () => {
    if (!selectedItem || !evaluation) return;

    try {
      const values = await reviewForm.validateFields();

      // 前端校验：反馈至少 2 字
      if (!values.feedback || values.feedback.trim().length < 2) {
        message.error('反馈意见至少需要 2 个字');
        return;
      }

      setSubmitting(true);
      setConflictError(null);

      // 构建分数调整（可选）
      const scoreAdjustments: ScoreAdjustments = {};
      const dimFields: (keyof ScoreAdjustments)[] = [
        'inquiry_score', 'knowledge_score', 'humanistic_score',
        'diagnosis_score', 'treatment_score',
      ];
      let hasAdjustments = false;
      const valuesRecord = values as Record<string, unknown>;
      for (const field of dimFields) {
        const val = valuesRecord[field];
        if (val !== undefined && val !== null && val !== '') {
          scoreAdjustments[field] = Number(val);
          hasAdjustments = true;
        }
      }

      await submitReview(evaluation.id, {
        feedback: values.feedback.trim(),
        score_adjustments: hasAdjustments ? scoreAdjustments : undefined,
      });

      message.success('复核已提交');
      setDetailModalOpen(false);
      reviewForm.resetFields();
      setEvaluation(null);
      setSelectedItem(null);
      // 刷新队列
      fetchQueue();
    } catch (err: unknown) {
      const error = err as { response?: { status?: number; data?: { error_code?: string; message?: string } } };
      if (error.response?.status === 409) {
        setConflictError('已被其他管理员复核');
        fetchQueue();
      } else if (error.response?.data?.message) {
        message.error(error.response.data.message);
      } else if (!(err as { errorFields?: unknown })?.errorFields) {
        // 非表单校验错误
        message.error('提交失败，请重试');
      }
    } finally {
      setSubmitting(false);
    }
  };

  const columns: ColumnsType<PendingReviewItem> = [
    { title: '问诊ID', dataIndex: 'consultation_id', key: 'consultation_id' },
    { title: '医生', dataIndex: 'doctor_username', key: 'doctor_username' },
    { title: '患者', dataIndex: 'patient_name', key: 'patient_name' },
    {
      title: '复核原因',
      dataIndex: 'review_reason',
      key: 'review_reason',
      ellipsis: true,
    },
    { title: '创建时间', dataIndex: 'created_at', key: 'created_at' },
    {
      title: '操作',
      key: 'action',
      render: (_: unknown, record: PendingReviewItem) => (
        <Button type="link" icon={<EyeOutlined />} onClick={() => handleViewDetail(record)}>
          查看
        </Button>
      ),
    },
  ];

  const renderRetryButton = () => (
    <Button icon={<ReloadOutlined />} onClick={fetchQueue}>
      重试
    </Button>
  );

  return (
    <div style={{ padding: 24 }}>
      <Title level={4}>
        <ExclamationCircleOutlined style={{ marginRight: 8, color: '#fa8c16' }} />
        人工复核工作台
      </Title>

      {/* 加载失败提示 */}
      {loadError && (
        <Alert
          type="error"
          showIcon
          message={loadError}
          action={renderRetryButton()}
          style={{ marginBottom: 16 }}
        />
      )}

      {/* 队列表格 */}
      <Table
        dataSource={queue}
        columns={columns}
        rowKey="evaluation_id"
        loading={loading}
        pagination={{ pageSize: 20 }}
      />

      {/* 复核详情弹窗 */}
      <Modal
        title="复核详情"
        open={detailModalOpen}
        onCancel={() => {
          setDetailModalOpen(false);
          reviewForm.resetFields();
          setEvaluation(null);
          setSelectedItem(null);
          setConflictError(null);
        }}
        width={800}
        footer={[
          <Button
            key="cancel"
            onClick={() => {
              setDetailModalOpen(false);
              reviewForm.resetFields();
              setEvaluation(null);
              setSelectedItem(null);
              setConflictError(null);
            }}
          >
            取消
          </Button>,
          <Button key="submit" type="primary" loading={submitting} onClick={handleReviewSubmit}>
            完成复核
          </Button>,
        ]}
      >
        {selectedItem && (
          <div>
            {/* 冲突错误提示 */}
            {conflictError && (
              <Alert
                type="warning"
                showIcon
                message={conflictError}
                style={{ marginBottom: 16 }}
              />
            )}

            {/* 基本信息 */}
            <Descriptions size="small" bordered column={2} style={{ marginBottom: 16 }}>
              <Descriptions.Item label="医生">{selectedItem.doctor_username}</Descriptions.Item>
              <Descriptions.Item label="患者">{selectedItem.patient_name}</Descriptions.Item>
              <Descriptions.Item label="复核原因" span={2}>{selectedItem.review_reason}</Descriptions.Item>
            </Descriptions>

            {detailLoading && <div style={{ textAlign: 'center', padding: 24 }}>加载中...</div>}

            {evaluation && !detailLoading && (
              <>
                {/* 检索/证据状态 */}
                <Descriptions size="small" bordered column={2} style={{ marginBottom: 16 }}>
                  <Descriptions.Item label="检索状态">
                    <Tag color={evaluation.retrieval_status === 'sufficient' ? 'green' : 'orange'}>
                      {evaluation.retrieval_status}
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="证据立场">
                    <Tag>{evaluation.evidence_stance}</Tag>
                  </Descriptions.Item>
                </Descriptions>

                {/* 五维原始评分 */}
                <Descriptions title="原始评分" size="small" bordered column={2} style={{ marginBottom: 16 }}>
                  <Descriptions.Item label="问诊">{evaluation.inquiry_score}</Descriptions.Item>
                  <Descriptions.Item label="知识">{evaluation.knowledge_score ?? 'N/A'}</Descriptions.Item>
                  <Descriptions.Item label="人文">{evaluation.humanistic_score}</Descriptions.Item>
                  <Descriptions.Item label="诊断">{evaluation.diagnosis_score}</Descriptions.Item>
                  <Descriptions.Item label="治疗">{evaluation.treatment_score}</Descriptions.Item>
                  <Descriptions.Item label="总分">{evaluation.total_score ?? 'N/A'}</Descriptions.Item>
                </Descriptions>

                {/* 引用 */}
                {evaluation.citation_data && evaluation.citation_data.length > 0 && (
                  <Card title="引用" size="small" style={{ marginBottom: 16 }}>
                    {evaluation.citation_data.map((cite, idx) => (
                      <div key={cite.citation_id || idx} style={{ padding: '8px 0', borderBottom: '1px solid #f0f0f0' }}>
                        <Space direction="vertical" size={0}>
                          <Text strong>{cite.claim}</Text>
                          <Text type="secondary">{cite.source} - {cite.heading_path}</Text>
                          <Text type="secondary" style={{ fontSize: 12 }}>{cite.text_snippet}</Text>
                        </Space>
                      </div>
                    ))}
                  </Card>
                )}

                {/* 复核表单 */}
                <Form form={reviewForm} layout="vertical">
                  <Form.Item
                    name="feedback"
                    label="反馈意见"
                    rules={[
                      { required: true, message: '必须填写反馈意见' },
                      { min: 2, message: '反馈意见至少 2 个字' },
                    ]}
                  >
                    <TextArea rows={3} placeholder="详细说明复核依据和调整理由..." />
                  </Form.Item>

                  {/* 可选五维分数调整 */}
                  <Card title="分数调整（可选）" size="small">
                    <Space wrap>
                      <Form.Item name="inquiry_score" label="问诊" style={{ marginBottom: 8 }}>
                        <Input type="number" min={0} max={100} style={{ width: 80 }} placeholder="0-100" />
                      </Form.Item>
                      <Form.Item name="knowledge_score" label="知识" style={{ marginBottom: 8 }}>
                        <Input type="number" min={0} max={100} style={{ width: 80 }} placeholder="0-100" />
                      </Form.Item>
                      <Form.Item name="humanistic_score" label="人文" style={{ marginBottom: 8 }}>
                        <Input type="number" min={0} max={100} style={{ width: 80 }} placeholder="0-100" />
                      </Form.Item>
                      <Form.Item name="diagnosis_score" label="诊断" style={{ marginBottom: 8 }}>
                        <Input type="number" min={0} max={100} style={{ width: 80 }} placeholder="0-100" />
                      </Form.Item>
                      <Form.Item name="treatment_score" label="治疗" style={{ marginBottom: 8 }}>
                        <Input type="number" min={0} max={100} style={{ width: 80 }} placeholder="0-100" />
                      </Form.Item>
                    </Space>
                  </Card>
                </Form>
              </>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default AdminReviews;
