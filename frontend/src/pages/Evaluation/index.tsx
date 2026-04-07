import React, { useEffect, useState, useCallback } from 'react';
import { Card, Typography, Progress, Row, Col, Descriptions, Button, Spin, message, Tag } from 'antd';
import { FileTextOutlined, RobotOutlined } from '@ant-design/icons';
import { useParams } from 'react-router-dom';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { getEvaluation, createEvaluation } from '../../api/evaluation';
import { getConsultationDetail } from '../../api/consultation';
import type { Evaluation, ConsultationDetail } from '../../types';

const { Title, Paragraph, Text } = Typography;

const scoreColor = (score: number) => {
  if (score >= 80) return '#52c41a';
  if (score >= 60) return '#faad14';
  return '#ff4d4f';
};

const EvaluationPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [consultation, setConsultation] = useState<ConsultationDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMsg, setProgressMsg] = useState('');
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    let timer: number;
    if (generating) {
      timer = window.setInterval(() => {
        setElapsed(prev => prev + 1);
      }, 1000);
    } else {
      setElapsed(0);
    }
    return () => clearInterval(timer);
  }, [generating]);

  const fetchData = useCallback(async () => {
    if (!id) return;
    setLoading(true);
    try {
      const detail = await getConsultationDetail(Number(id));
      setConsultation(detail);
    } catch {
      message.error('加载问诊详情失败');
    }
    try {
      const data = await getEvaluation(Number(id));
      setEvaluation(data);
    } catch {
      setEvaluation(null);
    }
    setLoading(false);
  }, [id]);

  const handleGenerate = async (isAutoRetry = false) => {
    if (!id) return;
    setGenerating(true);
    setProgress(0);
    setProgressMsg('正在初始化评估环境...');
    
    // 建立 WebSocket 连接以接收进度推送
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/evaluations/ws/${id}`;
    const ws = new WebSocket(wsUrl);
    
    // 设置 WebSocket 事件处理
    ws.onopen = () => {
      console.log('WebSocket connected for evaluation progress');
    };

    ws.onerror = (error) => {
      console.error('WebSocket connection error:', error);
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Progress update:', data);
        setProgress(data.progress);
        setProgressMsg(data.message);
      } catch (e) {
        console.error('Failed to parse WS message', e);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };

    // 等待 WebSocket 连接就绪（最多等待 3 秒）
    const waitForConnection = () => new Promise<void>((resolve) => {
      if (ws.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }
      const checkInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          clearInterval(checkInterval);
          resolve();
        }
      }, 100);
      // 3秒超时，即使连接未就绪也继续执行
      setTimeout(() => {
        clearInterval(checkInterval);
        resolve();
      }, 3000);
    });

    await waitForConnection();

    try {
      const data = await createEvaluation(Number(id));
      setEvaluation(data);
      message.success('评估报告已生成');
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: { error_type?: string } } } };
      const errorData = err.response?.data?.detail;
      if (errorData?.error_type === 'ValidationError' && !isAutoRetry) {
        message.warning('评估格式异常，正在尝试重新生成...');
        ws.close();
        handleGenerate(true);
        return;
      } else if (errorData?.error_type === 'ValidationError' && isAutoRetry) {
        message.error('评估格式异常，请稍后重试');
      } else {
        message.error('生成评估报告失败');
      }
    } finally {
      ws.close();
      setGenerating(false);
    }
  };

  useEffect(() => { fetchData(); }, [fetchData]);

  if (loading) return <Spin style={{ display: 'block', margin: '100px auto' }} size="large" />;

  if (generating) {
    return (
      <div style={{ textAlign: 'center', padding: '100px 0' }}>
        <RobotOutlined style={{ fontSize: 64, color: '#1677ff', marginBottom: 24 }} spin />
        <Title level={3}>AI 评估进行中</Title>
        <div style={{ maxWidth: 500, margin: '0 auto', padding: '0 24px' }}>
          <Progress percent={progress} status="active" strokeColor={{ '0%': '#108ee9', '100%': '#87d068' }} />
          <div style={{ marginTop: 16, fontSize: 16, color: '#666', fontWeight: 500 }}>{progressMsg}</div>
          {elapsed > 30 && (
            <div style={{ marginTop: 24, color: '#999', fontSize: 14 }}>
              后端正在深度分析中，请耐心等待... <br />
              预计剩余时间：约 {Math.max(5, 60 - elapsed)} 秒
            </div>
          )}
        </div>
      </div>
    );
  }

  if (!evaluation) {
    return (
      <div style={{ textAlign: 'center', padding: 80 }}>
        <RobotOutlined style={{ fontSize: 64, color: '#bbb', marginBottom: 24 }} />
        <Title level={4}>暂无评估报告</Title>
        <Paragraph type="secondary">点击下方按钮，七个 AI 智能体将协同分析您的问诊表现（五维评估 + 综合评分 + 改进建议）</Paragraph>
        {consultation?.diagnosis ? (
          <Card style={{ maxWidth: 500, margin: '16px auto', textAlign: 'left' }} size="small">
            <Text strong>已提交的诊断：</Text>
            <Paragraph style={{ margin: '4px 0 8px' }}>{consultation.diagnosis}</Paragraph>
            <Text strong>已提交的治疗方案：</Text>
            <Paragraph style={{ margin: '4px 0 0' }}>{consultation.treatment_plan}</Paragraph>
          </Card>
        ) : (
          <Paragraph type="warning">提示：您尚未提交诊断结果和治疗方案，相关维度评估可能不完整</Paragraph>
        )}
        <Button type="primary" size="large" onClick={() => handleGenerate()} loading={generating} icon={<FileTextOutlined />} style={{ marginTop: 16 }}>
          生成评估报告
        </Button>
      </div>
    );
  }

  const radarData = [
    { subject: '病史采集', score: evaluation.inquiry_score, fullMark: 100 },
    { subject: '医学知识', score: evaluation.knowledge_score, fullMark: 100 },
    { subject: '人文关怀', score: evaluation.humanistic_score, fullMark: 100 },
    { subject: '诊断结果', score: evaluation.diagnosis_score, fullMark: 100 },
    { subject: '治疗方案', score: evaluation.treatment_score, fullMark: 100 },
  ];

  return (
    <div>
      <Title level={4}>评估报告 <Tag color="success">问诊 #{evaluation.consultation_id}</Tag></Title>

      <Row gutter={16} style={{ marginBottom: 16 }}>
        {/* 综合评分与雷达图 */}
        <Col span={24}>
          <Card>
            <Row gutter={24} align="middle">
              <Col span={8} style={{ textAlign: 'center' }}>
                <Progress 
                  type="dashboard" 
                  percent={evaluation.total_score} 
                  size={160} 
                  strokeColor={scoreColor(evaluation.total_score)} 
                  format={(p) => <span style={{ fontSize: 36, fontWeight: 700 }}>{p}</span>} 
                />
                <div style={{ marginTop: 8, fontSize: 18, fontWeight: 600 }}>综合评分</div>
              </Col>
              <Col span={16} style={{ height: 320 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="subject" />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} />
                    <Tooltip />
                    <Radar
                      name="评估得分"
                      dataKey="score"
                      stroke="#1677ff"
                      fill="#1677ff"
                      fillOpacity={0.6}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      {/* 详细分析 */}
      <Card title="五维度详细分析" style={{ marginBottom: 16 }}>
        <Descriptions column={1} bordered size="small">
          <Descriptions.Item label="病史采集分析">
            <Paragraph style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{evaluation.inquiry_analysis}</Paragraph>
          </Descriptions.Item>
          <Descriptions.Item label="医学知识核对">
            <Paragraph style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{evaluation.knowledge_analysis}</Paragraph>
          </Descriptions.Item>
          <Descriptions.Item label="沟通交流评估">
            <Paragraph style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{evaluation.humanistic_analysis}</Paragraph>
          </Descriptions.Item>
          <Descriptions.Item label="诊断结果评估">
            <Paragraph style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{evaluation.diagnosis_analysis}</Paragraph>
          </Descriptions.Item>
          <Descriptions.Item label="治疗方案评估">
            <Paragraph style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{evaluation.treatment_analysis}</Paragraph>
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Card title="综合评价" style={{ marginBottom: 16 }}>
        <Paragraph style={{ whiteSpace: 'pre-wrap' }}>{evaluation.overall_summary}</Paragraph>
      </Card>

      <Card title="改进建议">
        <Paragraph style={{ whiteSpace: 'pre-wrap' }}>{evaluation.improvement_suggestions}</Paragraph>
      </Card>
    </div>
  );
};

export default EvaluationPage;
