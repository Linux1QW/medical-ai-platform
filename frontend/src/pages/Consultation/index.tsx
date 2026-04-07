import React, { useEffect, useRef, useState } from 'react';
import { Card, Input, Button, List, Avatar, Typography, Tag, Space, Spin, message, Modal, Form, Popconfirm, Divider, Progress } from 'antd';
import { SendOutlined, UserOutlined, MedicineBoxOutlined, FileTextOutlined, StopOutlined, PlusOutlined } from '@ant-design/icons';
import { useParams, useNavigate } from 'react-router-dom';
import { getConsultationDetail, sendMessage, submitDiagnosis, endConsultation, extendRounds } from '../../api/consultation';
import { getPatient } from '../../api/patient';
import type { Message, VirtualPatient } from '../../types';

const { Text, Title } = Typography;
const { TextArea } = Input;

const personalityColorMap: Record<string, string> = {
  配合型: 'green',
  焦虑型: 'orange',
  沉默型: 'blue',
  对抗型: 'red',
};

const ConsultationPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [messages, setMessages] = useState<Message[]>([]);
  const [patient, setPatient] = useState<VirtualPatient | null>(null);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [status, setStatus] = useState('in_progress');
  const [diagnosisVisible, setDiagnosisVisible] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [maxRounds, setMaxRounds] = useState(20);
  const [form] = Form.useForm();
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!id) return;
    getConsultationDetail(Number(id)).then((data) => {
      setMessages(data.messages);
      setStatus(data.status);
      setMaxRounds(data.max_rounds || 20);
      getPatient(data.patient_id).then(setPatient).catch(() => {
        message.error('加载患者信息失败');
      });
    }).catch(() => {
      message.error('加载问诊详情失败');
    });
  }, [id]);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || !id) return;
    setSending(true);
    try {
      const newMsgs = await sendMessage(Number(id), input.trim());
      setMessages((prev) => [...prev, ...newMsgs]);
      setInput('');
    } catch {
      message.error('发送消息失败');
    } finally {
      setSending(false);
    }
  };

  const handleEndConsultation = async () => {
    if (!id) return;
    try {
      await endConsultation(Number(id));
      setStatus('completed');
      message.success('问诊已结束');
    } catch {
      message.error('结束问诊失败');
    }
  };

  const handleExtendRounds = async () => {
    if (!id) return;
    try {
      const data = await extendRounds(Number(id));
      const newMax = data.max_rounds || maxRounds + 10;
      setMaxRounds(newMax);
      message.success(`已成功延长 10 轮问诊轮次（当前上限：${newMax}）`);
    } catch {
      message.error('延长轮次失败');
    }
  };

  const handleSubmitDiagnosis = async () => {
    if (!id) return;
    try {
      const values = await form.validateFields();
      setSubmitting(true);
      await submitDiagnosis(Number(id), values.diagnosis, values.treatment_plan);
      setStatus('completed');
      setDiagnosisVisible(false);
      message.success('诊断与治疗方案已提交，问诊结束');
    } catch {
      // 表单验证失败或API错误（API错误已由拦截器处理）
    } finally {
      setSubmitting(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const isEnded = status !== 'in_progress';
  const currentRounds = messages.filter(m => m.role === 'doctor').length;
  const isRoundLimitReached = currentRounds >= maxRounds;

  return (
    <div style={{ display: 'flex', gap: 16, height: 'calc(100vh - 180px)' }}>
      <Card style={{ width: 280, flexShrink: 0, overflow: 'auto' }} title="患者信息">
        {patient ? (
          <Space direction="vertical" size="small">
            <Text strong>姓名：{patient.name}</Text>
            <Text>年龄：{patient.age}岁</Text>
            <Text>性别：{patient.gender === 'male' ? '男' : '女'}</Text>
            <Text>主诉：{patient.chief_complaint}</Text>
            <Tag color={personalityColorMap[patient.personality_type] || 'blue'}>{patient.personality_type}</Tag>
            <Divider style={{ margin: '8px 0' }} />
            <Text type="secondary" style={{ fontSize: 12 }}>问诊轮次：{currentRounds} / {maxRounds}</Text>
            <Progress percent={parseFloat(Math.min(100, (currentRounds / maxRounds) * 100).toFixed(1))} size="small" status={isRoundLimitReached ? 'exception' : 'active'} />
          </Space>
        ) : (
          <Spin />
        )}
      </Card>

      <Card
        style={{ flex: 1, display: 'flex', flexDirection: 'column' }}
        title={
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Title level={5} style={{ margin: 0 }}>问诊对话</Title>
            <Space>
              <Tag color={isEnded ? 'default' : 'processing'}>{isEnded ? '已结束' : '进行中'}</Tag>
              {!isEnded && (
                <>
                  <Button type="primary" icon={<FileTextOutlined />} size="small" onClick={() => setDiagnosisVisible(true)}>
                    提交诊断
                  </Button>
                  <Popconfirm
                    title="确定要终止对话吗？终止后将无法继续问诊。"
                    onConfirm={handleEndConsultation}
                  >
                    <Button icon={<StopOutlined />} size="small">
                      终止对话
                    </Button>
                  </Popconfirm>
                </>
              )}
              {isEnded && (
                <Button type="primary" size="small" onClick={() => navigate(`/evaluation/${id}`)}>
                  查看评估
                </Button>
              )}
            </Space>
          </div>
        }
        styles={{ body: { flex: 1, display: 'flex', flexDirection: 'column', padding: 0, overflow: 'hidden' } }}
      >
        <div ref={listRef} style={{ flex: 1, overflow: 'auto', padding: '16px 24px' }}>
          <List
            dataSource={messages}
            renderItem={(msg) => (
              <List.Item style={{ border: 'none', justifyContent: msg.role === 'doctor' ? 'flex-end' : 'flex-start' }}>
                <div style={{ display: 'flex', gap: 8, maxWidth: '70%', flexDirection: msg.role === 'doctor' ? 'row-reverse' : 'row' }}>
                  <Avatar icon={msg.role === 'doctor' ? <MedicineBoxOutlined /> : <UserOutlined />}
                    style={{ backgroundColor: msg.role === 'doctor' ? '#1677ff' : '#87d068' }} />
                  <div style={{
                    padding: '8px 16px', borderRadius: 12,
                    background: msg.role === 'doctor' ? '#e6f4ff' : '#f6ffed',
                    whiteSpace: 'pre-wrap',
                  }}>
                    {msg.content}
                  </div>
                </div>
              </List.Item>
            )}
          />
          {sending && <div style={{ textAlign: 'center', padding: 16 }}><Spin tip="患者思考中..." /></div>}
        </div>

        {isRoundLimitReached && !isEnded && (
          <div style={{ padding: '8px 24px', background: '#fff2f0', borderTop: '1px solid #ffccc7', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Text type="danger">已达到最大问诊轮次（{maxRounds} 轮），请提交诊断以评估或延长轮次。</Text>
            <Button size="small" type="primary" icon={<PlusOutlined />} onClick={handleExtendRounds}>延长 10 轮</Button>
          </div>
        )}

        {!isEnded && (
          <div style={{ padding: 16, borderTop: '1px solid #f0f0f0', display: 'flex', gap: 8 }}>
            <Input.TextArea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={isRoundLimitReached ? "已达到轮次上限，请先延长或提交评估" : "输入您的问诊内容...（Enter 发送）"}
              autoSize={{ minRows: 1, maxRows: 4 }}
              disabled={sending || isRoundLimitReached}
            />
            <Button type="primary" icon={<SendOutlined />} onClick={handleSend} loading={sending} disabled={isRoundLimitReached}>
              发送
            </Button>
          </div>
        )}
      </Card>

      <Modal
        title="提交诊断结果与治疗方案"
        open={diagnosisVisible}
        onCancel={() => setDiagnosisVisible(false)}
        onOk={handleSubmitDiagnosis}
        confirmLoading={submitting}
        okText="提交并结束问诊"
        cancelText="继续问诊"
        width={640}
      >
        <Form form={form} layout="vertical">
          <Form.Item name="diagnosis" label="诊断结果" rules={[{ required: true, message: '请输入诊断结果' }]}>
            <TextArea rows={4} placeholder="请输入您的诊断结果，包括主诊断和鉴别诊断..." />
          </Form.Item>
          <Form.Item name="treatment_plan" label="治疗方案" rules={[{ required: true, message: '请输入治疗方案' }]}>
            <TextArea rows={4} placeholder="请输入治疗方案，包括药物治疗、非药物治疗、随访计划等..." />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default ConsultationPage;
