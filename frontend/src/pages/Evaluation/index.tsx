import React, { useEffect, useState, useCallback, useRef } from 'react';
import { Card, Typography, Progress, Row, Col, Collapse, Button, Spin, message, Tag, Badge } from 'antd';
import { FileTextOutlined, RobotOutlined, MedicineBoxOutlined, ReadOutlined, HeartOutlined, CheckCircleOutlined, ExperimentOutlined, BulbOutlined, TrophyOutlined, WarningOutlined, ToolOutlined } from '@ant-design/icons';
import { useParams } from 'react-router-dom';
import { getEvaluation, createEvaluation, getEvaluationLockStatus } from '../../api/evaluation';
import { getConsultationDetail } from '../../api/consultation';
import type { Evaluation, ConsultationDetail, Citation } from '../../types';
import { ScoreDisplay, DimensionRadar, getScoreColor, getScoreLevel } from '../../components';

const { Title, Paragraph, Text } = Typography;

const EvaluationPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [consultation, setConsultation] = useState<ConsultationDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMsg, setProgressMsg] = useState('');
  const [elapsed, setElapsed] = useState(0);
  const [lockActive, setLockActive] = useState(false);
  const pollingRef = useRef<number | null>(null);

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

  useEffect(() => { fetchData(); }, [fetchData]);

  // 页面加载时检查是否有进行中的评估
  useEffect(() => {
    if (!id) return;
    let cancelled = false;

    const checkLock = async () => {
      try {
        const status = await getEvaluationLockStatus(Number(id));
        if (!cancelled && status.is_active && !evaluation) {
          setLockActive(true);
          setGenerating(true);
          startPolling();
        }
      } catch {
        // 接口不可用时降级
      }
    };

    if (!evaluation) {
      checkLock();
    }

    return () => {
      cancelled = true;
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [id]);

  const startPolling = () => {
    if (pollingRef.current) clearInterval(pollingRef.current);
    pollingRef.current = window.setInterval(async () => {
      try {
        const data = await getEvaluation(Number(id));
        if (data) {
          setEvaluation(data);
          setGenerating(false);
          setLockActive(false);
          if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
          }
          message.success('评估报告已生成');
        }
      } catch {
        // 评估尚未完成
      }
    }, 5000);
  };

  const handleGenerate = async (isAutoRetry = false) => {
    if (!id) return;
    setGenerating(true);
    setProgress(0);
    setProgressMsg('正在初始化评估环境...');
    
    // 建立 WebSocket 连接以接收进度推送
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const token = sessionStorage.getItem('token') || '';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/evaluations/ws/${id}?token=${encodeURIComponent(token)}`;
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
      const data = (error as { response?: { data?: { error_type?: string } } })?.response?.data;
      if (data?.error_type === 'ValidationError' && !isAutoRetry) {
        message.warning('评估格式异常，正在尝试重新生成...');
        ws.close();
        handleGenerate(true);
        return;
      } else if (data?.error_type === 'ValidationError' && isAutoRetry) {
        message.error('评估格式异常，请稍后重试');
      } else {
        message.error('生成评估报告失败');
      }
    } finally {
      ws.close();
      setGenerating(false);
      setLockActive(false);
    }
  };

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
        <Paragraph type="secondary">点击按钮后，七个 AI 智能体将协同分析您的问诊表现（五维评估 + 综合评分 + 改进建议）</Paragraph>
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
        {!lockActive && (
          <Button type="primary" size="large" onClick={() => handleGenerate()} icon={<FileTextOutlined />} style={{ marginTop: 16 }} loading={generating}>
            生成评估报告
          </Button>
        )}
        {lockActive && (
          <div style={{ marginTop: 16, textAlign: 'center' }}>
            <Spin size="large" />
            <div style={{ marginTop: 12, color: '#666', fontSize: 15 }}>
              评估正在进行中，请稍候...
            </div>
          </div>
        )}
      </div>
    );
  }

  const isNeedsReview = evaluation.evaluation_status === 'needs_review' || evaluation.human_review_needed === true;

  const radarScores = {
    病史采集: evaluation.inquiry_score,
    医学知识: isNeedsReview ? null : (evaluation.knowledge_score ?? null),
    人文关怀: evaluation.humanistic_score,
    诊断结果: evaluation.diagnosis_score,
    治疗方案: evaluation.treatment_score,
  };

  const dimensionItems = [
    { key: 'inquiry', label: '病史采集', score: evaluation.inquiry_score, analysis: evaluation.inquiry_analysis, icon: <ReadOutlined /> },
    { key: 'diagnosis', label: '诊断结果', score: evaluation.diagnosis_score, analysis: evaluation.diagnosis_analysis, icon: <CheckCircleOutlined /> },
    { key: 'treatment', label: '治疗方案', score: evaluation.treatment_score, analysis: evaluation.treatment_analysis, icon: <MedicineBoxOutlined /> },
    { key: 'knowledge', label: '医学知识', score: evaluation.knowledge_score, analysis: evaluation.knowledge_analysis, icon: <ExperimentOutlined />, isReview: isNeedsReview },
    { key: 'humanistic', label: '人文关怀', score: evaluation.humanistic_score, analysis: evaluation.humanistic_analysis, icon: <HeartOutlined /> },
  ];


  const parseSuggestions = (text: string) => {
    if (!text) return [];
    const sections = text.split(/(?=[一二三四五六七八九十]、)/);
    return sections.filter(s => s.trim()).map((section) => {
      const titleMatch = section.match(/^([一二三四五六七八九十])、(.+?)(?:\n|$)/);
      const number = titleMatch ? titleMatch[1] : '';
      const title = titleMatch ? titleMatch[2].trim() : '';
      const problemMatch = section.match(/问题描述[：:]\s*([\s\S]*?)(?=\n*改进方法[：:]|$)/);
      const problem = problemMatch ? problemMatch[1].trim() : '';
      const methodMatch = section.match(/改进方法[：:]\s*([\s\S]*?)$/);
      const method = methodMatch ? methodMatch[1].trim() : '';
      // 如果没有匹配到"问题描述/改进方法"结构，将标题后的全部内容作为 content
      let content = '';
      if (!problem && !method && titleMatch) {
        content = section.replace(/^[一二三四五六七八九十]、.+?\n+/, '').trim();
      }
      return { number, title, problem, method, content };
    });
  };

  const suggestionItems = parseSuggestions(evaluation.improvement_suggestions || '');
  const hasParsedSuggestions = suggestionItems.length > 0 && suggestionItems.some(s => s.title);
  const suggestionColors = ['#1677ff', '#52c41a', '#faad14', '#722ed1', '#eb2f96'];

  return (
    <div>
      <Title level={4}>评估报告 <Tag color="success">问诊 #{evaluation.consultation_id}</Tag></Title>

      <Row gutter={16} style={{ marginBottom: 16 }}>
        {/* 综合评分与雷达图 */}
        <Col span={24}>
          <Card>
            <Row gutter={24} align="middle">
              <Col span={8} style={{ textAlign: 'center' }}>
                {isNeedsReview ? (
                  <div>
                    <Tag color="orange" style={{ fontSize: 20, padding: '8px 24px' }}>证据不足</Tag>
                    <div style={{ marginTop: 8, fontSize: 18, fontWeight: 600 }}>综合评分（待复核）</div>
                  </div>
                ) : (
                  <>
                    <Progress 
                      type="dashboard" 
                      percent={evaluation.total_score ?? 0} 
                      size={160} 
                      strokeColor={getScoreColor(evaluation.total_score ?? 0)} 
                      format={(p) => <span style={{ fontSize: 36, fontWeight: 700 }}>{p}</span>} 
                    />
                    <div style={{ marginTop: 8, fontSize: 18, fontWeight: 600 }}>综合评分</div>
                  </>
                )}
              </Col>
              <Col span={16} style={{ height: 320 }}>
                <DimensionRadar scores={radarScores} />
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      {/* 五维度分数概览卡片 */}
      <Row gutter={[12, 12]} style={{ marginBottom: 16 }}>
        {dimensionItems.map(item => {
          const level = item.isReview ? { text: '待复核', color: '#fa8c16' } : getScoreLevel(item.score ?? 0);
          return (
            <Col xs={12} sm={8} lg={Math.floor(24 / 5)} key={item.key} style={{ minWidth: 140 }}>
              <Card size="small" style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 20, color: '#555', marginBottom: 4 }}>{item.icon}</div>
                <div style={{ fontSize: 13, color: '#888' }}>{item.label}</div>
                <div style={{ fontSize: 28, fontWeight: 700, color: level.color }}>
                  {item.isReview ? <Tag color="orange">待复核</Tag> : <ScoreDisplay score={item.score ?? 0} size="large" />}
                </div>
                <Tag color={level.color} style={{ borderRadius: 10 }}>{level.text}</Tag>
              </Card>
            </Col>
          );
        })}
      </Row>

      {/* 详细分析 - 可折叠面板 */}
      <Card title="五维度详细分析" style={{ marginBottom: 16 }}>
        <Collapse
          accordion
          items={dimensionItems.map(item => {
            const level = item.isReview ? { text: '待复核', color: '#fa8c16' } : getScoreLevel(item.score ?? 0);
            return {
              key: item.key,
              label: (
                <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  {item.icon}
                  <span style={{ fontWeight: 500 }}>{item.label}</span>
                  {item.isReview ? (
                    <Tag color="orange">待复核</Tag>
                  ) : (
                    <Badge
                      count={item.score ?? 0}
                      showZero
                      style={{ backgroundColor: level.color, fontWeight: 600, fontSize: 13 }}
                      overflowCount={100}
                    />
                  )}
                </span>
              ),
              children: (
                <Paragraph style={{ whiteSpace: 'pre-wrap', margin: 0, lineHeight: 1.8, color: '#333' }}>
                  {item.analysis}
                </Paragraph>
              ),
            };
          })}
        />
      </Card>

      {/* 证据详情 - 仅在有需要复核或引用数据时显示 */}
      {(isNeedsReview || evaluation.citation_data?.length) && (
        <Card
          title={<span><ExperimentOutlined style={{ color: '#1677ff', marginRight: 8 }} />医学证据详情</span>}
          style={{ marginBottom: 16 }}
        >
          <Row gutter={16} style={{ marginBottom: 12 }}>
            <Col span={8}>
              <Text type="secondary">检索状态：</Text>
              <Tag color={
                evaluation.retrieval_status === 'sufficient' ? 'green' :
                evaluation.retrieval_status === 'insufficient' ? 'orange' :
                evaluation.retrieval_status === 'error' ? 'red' : 'default'
              }>{evaluation.retrieval_status}</Tag>
            </Col>
            <Col span={8}>
              <Text type="secondary">证据立场：</Text>
              <Tag color={
                evaluation.evidence_stance === 'supports' ? 'green' :
                evaluation.evidence_stance === 'contradicts' ? 'red' :
                evaluation.evidence_stance === 'mixed' ? 'orange' : 'default'
              }>{evaluation.evidence_stance}</Tag>
            </Col>
            <Col span={8}>
              <Text type="secondary">人工复核：</Text>
              <Tag color={evaluation.human_review_needed ? 'orange' : 'green'}>
                {evaluation.human_review_needed ? '需要' : '不需要'}
              </Tag>
            </Col>
          </Row>
          {evaluation.review_reason && (
            <div style={{ padding: '8px 14px', background: '#fff7e6', borderRadius: 4, marginBottom: 12 }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                <WarningOutlined style={{ color: '#faad14', marginRight: 4 }} />复核原因：
              </Text>
              <Text>{evaluation.review_reason}</Text>
            </div>
          )}
          {evaluation.citation_data && evaluation.citation_data.length > 0 && (
            <div>
              <Text strong style={{ display: 'block', marginBottom: 8 }}>引用文献（{evaluation.citation_data.length} 条）：</Text>
              {evaluation.citation_data?.map((cite: Citation) => (
                <div key={cite.citation_id || cite.source + cite.page} style={{
                  borderLeft: '3px solid #1677ff', padding: '8px 12px', marginBottom: 8,
                  background: '#fafafa', borderRadius: 4,
                }}>
                  {cite.claim && <div style={{ fontSize: 13, marginBottom: 4, color: '#333' }}><strong>结论：</strong>{cite.claim}</div>}
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <Text strong style={{ fontSize: 13 }}>{cite.source}</Text>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {cite.page ? `第 ${cite.page} 页` : ''}
                      {cite.rerank_score != null ? ` | 综合重排分 ${(cite.rerank_score * 100).toFixed(0)}%` : ''}
                    </Text>
                  </div>
                  {cite.heading_path && <Text type="secondary" style={{ fontSize: 12, display: 'block' }}>{cite.heading_path}</Text>}
                  <Paragraph style={{ fontSize: 13, margin: '4px 0 0', lineHeight: 1.6, color: '#555' }} ellipsis={{ rows: 3 }}>
                    {cite.text_snippet}
                  </Paragraph>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}

      {/* 综合评价 */}
      <Card
        title={<span><TrophyOutlined style={{ color: '#faad14', marginRight: 8 }} />综合评价</span>}
        style={{ marginBottom: 16 }}
      >
        <Paragraph style={{ whiteSpace: 'pre-wrap', lineHeight: 1.8 }}>{evaluation.overall_summary}</Paragraph>
      </Card>

      {/* 改进建议 - 结构化展示 */}
      <Card
        title={<span><BulbOutlined style={{ color: '#1677ff', marginRight: 8 }} />改进建议</span>}
        style={{ marginBottom: 16 }}
      >
        {hasParsedSuggestions ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {suggestionItems.map((item, idx) => (
              <div
                key={idx}
                style={{
                  borderLeft: `4px solid ${suggestionColors[idx % suggestionColors.length]}`,
                  borderRadius: 6,
                  background: '#fafafa',
                  padding: '16px 20px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                  <span style={{
                    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                    width: 28, height: 28, borderRadius: '50%',
                    background: suggestionColors[idx % suggestionColors.length],
                    color: '#fff', fontWeight: 700, fontSize: 14,
                  }}>
                    {item.number}
                  </span>
                  <Text strong style={{ fontSize: 15 }}>{item.title}</Text>
                </div>
                {item.problem && (
                  <div style={{ marginBottom: 10, padding: '10px 14px', background: '#fff7e6', borderRadius: 4 }}>
                    <Text type="secondary" style={{ fontSize: 12, display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4 }}>
                      <WarningOutlined style={{ color: '#faad14' }} />问题描述
                    </Text>
                    <Text style={{ lineHeight: 1.8 }}>{item.problem}</Text>
                  </div>
                )}
                {item.method && (
                  <div style={{ padding: '10px 14px', background: '#e6f7ff', borderRadius: 4 }}>
                    <Text type="secondary" style={{ fontSize: 12, display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4 }}>
                      <ToolOutlined style={{ color: '#1677ff' }} />改进方法
                    </Text>
                    <Text style={{ lineHeight: 1.8 }}>{item.method}</Text>
                  </div>
                )}
                {item.content && !item.problem && !item.method && (
                  <div style={{ padding: '10px 14px', background: '#f6ffed', borderRadius: 4 }}>
                    <Text style={{ lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>{item.content}</Text>
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <Paragraph style={{ whiteSpace: 'pre-wrap', lineHeight: 1.8 }}>{evaluation.improvement_suggestions}</Paragraph>
        )}
      </Card>
    </div>
  );
};

export default EvaluationPage;
