import React, { useEffect, useState } from 'react';
import { Card, Col, Row, Statistic, Typography, List, Tag, Button, Select, Space } from 'antd';
import {
  MessageOutlined,
  TeamOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  LineChartOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import dayjs from 'dayjs';
import { getConsultations } from '../../api/consultation';
import { useAuth } from '../../store/useAuth';
import type { Consultation } from '../../types';

const { Title, Text } = Typography;

const statusMap: Record<string, { color: string; text: string }> = {
  in_progress: { color: 'processing', text: '进行中' },
  completed: { color: 'warning', text: '已完成' },
  evaluated: { color: 'success', text: '已评估' },
};

const DashboardPage: React.FC = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [consultations, setConsultations] = useState<Consultation[]>([]);
  const [days, setDays] = useState(30);

  useEffect(() => {
    getConsultations().then(setConsultations).catch(() => {});
  }, []);

  const inProgress = consultations.filter((c) => c.status === 'in_progress').length;
  const completed = consultations.filter((c) => c.status !== 'in_progress').length;

  // 处理趋势数据
  const trendData = consultations
    .filter(c => c.status === 'evaluated' && c.total_score !== undefined && c.total_score !== null)
    .filter(c => dayjs(c.started_at).isAfter(dayjs().subtract(days, 'days')))
    .map(c => ({
      id: c.id,
      time: dayjs(c.started_at).format('MM-DD'),
      score: c.total_score,
      fullTime: dayjs(c.started_at).format('YYYY-MM-DD HH:mm')
    }))
    .reverse(); // 按时间正序排列

  const avgScore = trendData.length > 0 
    ? Math.round(trendData.reduce((acc, curr) => acc + (curr.score || 0), 0) / trendData.length) 
    : 0;

  return (
    <div>
      <Title level={4}>欢迎回来，{user?.real_name || user?.username}</Title>
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic title="总问诊数" value={consultations.length} prefix={<MessageOutlined />} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="进行中" value={inProgress} prefix={<ClockCircleOutlined />} valueStyle={{ color: '#1677ff' }} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="已完成" value={completed} prefix={<CheckCircleOutlined />} valueStyle={{ color: '#52c41a' }} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Button type="primary" icon={<TeamOutlined />} size="large" block onClick={() => navigate('/patients')}>
              开始新问诊
            </Button>
          </Card>
        </Col>
      </Row>

      <Row gutter={16}>
        <Col span={16}>
          <Card 
            title={<span><LineChartOutlined style={{ marginRight: 8 }} />历次评分趋势</span>} 
            style={{ marginBottom: 24 }}
            extra={
              <Select value={days} onChange={setDays} style={{ width: 120 }} size="small">
                <Select.Option value={30}>近 30 天</Select.Option>
                <Select.Option value={90}>近 90 天</Select.Option>
                <Select.Option value={180}>近 180 天</Select.Option>
              </Select>
            }
          >
            <div style={{ height: 350 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart 
                  data={trendData} 
                  onClick={(data: any) => {
                    if (data?.activePayload) {
                      navigate(`/evaluation/${data.activePayload[0].payload.id}`);
                    }
                  }}
                  margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="time" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip 
                    labelFormatter={(_v, items: any) => items[0]?.payload?.fullTime}
                    contentStyle={{ borderRadius: 8, border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                  />
                  <ReferenceLine y={80} label={{ position: 'right', value: '目标线(80)', fill: '#ff4d4f', fontSize: 12 }} stroke="#ff4d4f" strokeDasharray="3 3" />
                  <ReferenceLine y={avgScore} label={{ position: 'left', value: `平均线(${avgScore})`, fill: '#52c41a', fontSize: 12 }} stroke="#52c41a" strokeDasharray="3 3" />
                  <Line 
                    type="monotone" 
                    dataKey="score" 
                    stroke="#1677ff" 
                    strokeWidth={3}
                    dot={{ r: 4, fill: '#1677ff' }}
                    activeDot={{ r: 8, cursor: 'pointer' }} 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </Col>
        <Col span={8}>
          <Card title="最近问诊记录" styles={{ body: { padding: '0 16px' } }}>
            <List
              dataSource={consultations.slice(0, 6)}
              renderItem={(item) => (
                <List.Item
                  actions={[
                    <Button type="link" size="small" onClick={() => navigate(`/consultation/${item.id}`)}>
                      查看
                    </Button>,
                  ]}
                >
                  <List.Item.Meta
                    title={
                      <Space>
                        <Text strong>问诊 #{item.id}</Text>
                        <Tag color={statusMap[item.status]?.color} style={{ fontSize: 10, lineHeight: '16px' }}>{statusMap[item.status]?.text}</Tag>
                      </Space>
                    }
                    description={dayjs(item.started_at).format('MM-DD HH:mm')}
                  />
                </List.Item>
              )}
              locale={{ emptyText: '暂无记录' }}
            />
            {consultations.length > 6 && (
              <div style={{ textAlign: 'center', padding: '12px 0' }}>
                <Button type="link" onClick={() => navigate('/consultations')}>查看全部</Button>
              </div>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default DashboardPage;
