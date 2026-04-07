import React, { useEffect, useState } from 'react';
import { Card, Table, Tag, Typography, Button, Space, Input, Form, Select, DatePicker, Row, Col } from 'antd';
import { EyeOutlined, FileTextOutlined, SearchOutlined, ClearOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import dayjs from 'dayjs';
import { getAllConsultations } from '../../api/consultation';
import type { Consultation } from '../../types';

const { Title } = Typography;

const personalityMap: Record<string, { color: string; text: string }> = {
  配合型: { color: 'green', text: '配合型' },
  焦虑型: { color: 'orange', text: '焦虑型' },
  沉默型: { color: 'blue', text: '沉默型' },
  对抗型: { color: 'red', text: '对抗型' },
};

const statusMap: Record<string, { color: string; text: string }> = {
  in_progress: { color: 'processing', text: '进行中' },
  completed: { color: 'warning', text: '已完成' },
  evaluated: { color: 'success', text: '已评估' },
};

const AdminConsultationsPage: React.FC = () => {
  const navigate = useNavigate();
  const [form] = Form.useForm();
  const [consultations, setConsultations] = useState<Consultation[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchList = (values: any = {}) => {
    setLoading(true);
    const params: any = { ...values };
    if (values.timeRange) {
      params.start_time = values.timeRange[0].startOf('day').toISOString();
      params.end_time = values.timeRange[1].endOf('day').toISOString();
      delete params.timeRange;
    }
    getAllConsultations(params)
      .then(setConsultations)
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchList();
  }, []);

  const onFinish = (values: any) => {
    fetchList(values);
  };

  const onReset = () => {
    form.resetFields();
    fetchList();
  };

  const columns = [
    { title: 'ID', dataIndex: 'id', key: 'id', width: 70 },
    { title: '问诊医生', dataIndex: 'doctor_username', key: 'doctor_username', width: 120 },
    { title: '患者姓名', dataIndex: 'patient_name', key: 'patient_name' },
    {
      title: '人格类型', dataIndex: 'personality_type', key: 'personality_type', width: 100,
      render: (v: string) => {
        const item = personalityMap[v] || { color: 'default', text: v };
        return <Tag color={item.color}>{item.text}</Tag>;
      },
    },
    {
      title: '最终评分', dataIndex: 'total_score', key: 'total_score', sorter: (a: Consultation, b: Consultation) => (a.total_score || 0) - (b.total_score || 0),
      render: (v: number | null) => (v !== null ? <Tag color={v >= 80 ? 'green' : v >= 60 ? 'orange' : 'red'}>{v}</Tag> : '-'),
    },
    {
      title: '状态', dataIndex: 'status', key: 'status', width: 100,
      render: (v: string) => <Tag color={statusMap[v]?.color}>{statusMap[v]?.text}</Tag>,
    },
    {
      title: '开始时间', dataIndex: 'started_at', key: 'started_at', sorter: (a: Consultation, b: Consultation) => new Date(a.started_at).getTime() - new Date(b.started_at).getTime(),
      render: (v: string) => dayjs(v).format('YYYY-MM-DD HH:mm'),
    },
    {
      title: '操作', key: 'action', width: 180,
      render: (_: unknown, record: Consultation) => (
        <Space>
          <Button type="link" size="small" icon={<EyeOutlined />} onClick={() => navigate(`/consultation/${record.id}`)}>
            对话
          </Button>
          {record.status !== 'in_progress' && (
            <Button type="link" size="small" icon={<FileTextOutlined />} onClick={() => navigate(`/evaluation/${record.id}`)}>
              报告
            </Button>
          )}
        </Space>
      ),
    },
  ];

  return (
    <div>
      <Title level={4} style={{ marginBottom: 16 }}>全平台问诊记录</Title>
      
      <Card style={{ marginBottom: 16 }}>
        <Form form={form} layout="inline" onFinish={onFinish}>
          <Row gutter={[16, 16]} style={{ width: '100%' }}>
            <Col>
              <Form.Item name="username" label="医生用户名">
                <Input placeholder="搜索用户名" allowClear style={{ width: 150 }} />
              </Form.Item>
            </Col>
            <Col>
              <Form.Item name="personality" label="患者人格">
                <Select placeholder="选择人格" allowClear style={{ width: 120 }}>
                  <Select.Option value="配合型">配合型</Select.Option>
                  <Select.Option value="焦虑型">焦虑型</Select.Option>
                  <Select.Option value="沉默型">沉默型</Select.Option>
                  <Select.Option value="对抗型">对抗型</Select.Option>
                </Select>
              </Form.Item>
            </Col>
            <Col>
              <Form.Item name="timeRange" label="时间范围">
                <DatePicker.RangePicker />
              </Form.Item>
            </Col>
            <Col>
              <Space>
                <Button type="primary" icon={<SearchOutlined />} htmlType="submit">查询</Button>
                <Button icon={<ClearOutlined />} onClick={onReset}>重置</Button>
              </Space>
            </Col>
          </Row>
        </Form>
      </Card>

      <Card>
        <Table dataSource={consultations} columns={columns} rowKey="id" loading={loading} pagination={{ pageSize: 10 }} />
      </Card>
    </div>
  );
};

export default AdminConsultationsPage;