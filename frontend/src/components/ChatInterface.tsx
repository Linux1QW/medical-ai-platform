import React, { useState, useRef, useEffect } from 'react';
import { Card, Input, Button, List, Avatar, Typography, Tag, Space } from 'antd';
import { SendOutlined, UserOutlined, MedicineBoxOutlined, LoadingOutlined } from '@ant-design/icons';
import type { Message, VirtualPatient } from '../types';

const { Text, Title } = Typography;
const { TextArea } = Input;

interface ChatInterfaceProps {
  consultationId: number;
  patient: VirtualPatient | null;
  messages: Message[];
  status: string;
  sending: boolean;
  onSendMessage: (content: string) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  patient,
  messages,
  status,
  sending,
  onSendMessage,
}) => {
  const [input, setInput] = useState('');
  const messagesRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;
    onSendMessage(input.trim());
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const isEnded = status !== 'in_progress';
  const currentRounds = messages.filter(m => m.role === 'doctor').length;

  return (
    <Card
      style={{ flex: 1, display: 'flex', flexDirection: 'column' }}
      title={
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Title level={5} style={{ margin: 0 }}>问诊对话</Title>
          <Space>
            <Tag color={isEnded ? 'default' : 'processing'}>
              {isEnded ? '已结束' : '进行中'}
            </Tag>
            {patient && (
              <Text type="secondary" style={{ fontSize: 12 }}>
                轮次：{currentRounds}
              </Text>
            )}
          </Space>
        </div>
      }
      styles={{ body: { flex: 1, display: 'flex', flexDirection: 'column', padding: 0, overflow: 'hidden' } }}
    >
      <div
        ref={messagesRef}
        style={{ flex: 1, overflow: 'auto', padding: '16px 24px' }}
      >
        <List
          dataSource={messages}
          renderItem={(msg) => (
            <List.Item
              style={{
                border: 'none',
                justifyContent: msg.role === 'doctor' ? 'flex-end' : 'flex-start',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  gap: 8,
                  maxWidth: '70%',
                  flexDirection: msg.role === 'doctor' ? 'row-reverse' : 'row',
                }}
              >
                <Avatar
                  icon={msg.role === 'doctor' ? <MedicineBoxOutlined /> : <UserOutlined />}
                  style={{ backgroundColor: msg.role === 'doctor' ? '#1677ff' : '#87d068' }}
                />
                <div
                  style={{
                    padding: '8px 16px',
                    borderRadius: 12,
                    background: msg.role === 'doctor' ? '#e6f4ff' : '#f6ffed',
                    whiteSpace: 'pre-wrap',
                  }}
                >
                  {msg.content}
                </div>
              </div>
            </List.Item>
          )}
        />
        {sending && (
          <div style={{ padding: '8px 24px', color: '#666', display: 'flex', alignItems: 'center', gap: 8 }}>
            <LoadingOutlined />
            <span>患者思考中...</span>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {!isEnded && (
        <div style={{ padding: 16, borderTop: '1px solid #f0f0f0', display: 'flex', gap: 8 }}>
          <TextArea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入您的问诊内容...（Enter 发送）"
            autoSize={{ minRows: 1, maxRows: 4 }}
            disabled={sending}
          />
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSend}
            loading={sending}
          >
            发送
          </Button>
        </div>
      )}
    </Card>
  );
};

export default ChatInterface;
