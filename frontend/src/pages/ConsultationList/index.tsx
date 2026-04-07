import React, { useEffect, useState } from 'react';
import { Card, Table, Tag, Typography, Button, Space, Popconfirm, message, Input } from 'antd';
import { EyeOutlined, FileTextOutlined, DeleteOutlined, SearchOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { getConsultations, deleteConsultation } from '../../api/consultation';
import type { Consultation } from '../../types';

const { Title } = Typography;

const statusMap: Record<string, { color: string; text: string }> = {
  in_progress: { color: 'processing', text: '进行中' },
  completed: { color: 'warning', text: '已完成' },
  evaluated: { color: 'success', text: '已评估' },
};

const ConsultationListPage: React.FC = () => {
  const navigate = useNavigate();
  const [consultations, setConsultations] = useState<Consultation[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchText, setSearchText] = useState('');

  const fetchList = () => {
    setLoading(true);
    getConsultations()
      .then(setConsultations)
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchList();
  }, []);

  const handleDelete = async (id: number) => {
    try {
      await deleteConsultation(id);
      message.success('已删除');
      fetchList();
    } catch {
      // error already shown by request
    }
  };

  const filteredData = consultations.filter(item => 
    item.patient_name?.toLowerCase().includes(searchText.toLowerCase()) ||
    item.personality_type?.toLowerCase().includes(searchText.toLowerCase()) ||
    item.id.toString().includes(searchText)
  );

  const columns = [
    {
      title: '序号',
      key: 'index',
      width: 70,
      render: (_: unknown, __: Consultation, index: number) => index + 1,
    },
    { title: '患者姓名', dataIndex: 'patient_name', key: 'patient_name', sorter: (a: Consultation, b: Consultation) => (a.patient_name || '').localeCompare(b.patient_name || '') },
    { title: '人格类型', dataIndex: 'personality_type', key: 'personality_type' },
    {
      title: '最终评分', dataIndex: 'total_score', key: 'total_score', sorter: (a: Consultation, b: Consultation) => (a.total_score || 0) - (b.total_score || 0),
      render: (v: number | null) => (v !== null ? <Tag color={v >= 80 ? 'green' : v >= 60 ? 'orange' : 'red'}>{v}</Tag> : '-'),
    },
    {
      title: '用时(分)', dataIndex: 'duration_minutes', key: 'duration_minutes', sorter: (a: Consultation, b: Consultation) => (a.duration_minutes || 0) - (b.duration_minutes || 0),
      render: (v: number | null) => (v !== null ? `${v} 分钟` : '-'),
    },
    {
      title: '状态', dataIndex: 'status', key: 'status', width: 100,
      render: (v: string) => <Tag color={statusMap[v]?.color}>{statusMap[v]?.text}</Tag>,
    },
    {
      title: '开始时间', dataIndex: 'started_at', key: 'started_at', sorter: (a: Consultation, b: Consultation) => new Date(a.started_at).getTime() - new Date(b.started_at).getTime(),
      render: (v: string) => new Date(v).toLocaleString(),
    },
    {
      title: '操作', key: 'action', width: 220,
      render: (_: unknown, record: Consultation) => (
        <Space>
          <Button type="link" size="small" icon={<EyeOutlined />} onClick={() => navigate(`/consultation/${record.id}`)}>
            对话
          </Button>
          {record.status !== 'in_progress' && (
            <Button type="link" size="small" icon={<FileTextOutlined />} onClick={() => navigate(`/evaluation/${record.id}`)}>
              评估
            </Button>
          )}
          <Popconfirm
            title="确定删除该问诊记录？"
            onConfirm={() => handleDelete(record.id)}
          >
            <Button type="link" size="small" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>问诊记录</Title>
        <Input
          placeholder="搜索患者姓名、人格类型..."
          prefix={<SearchOutlined />}
          onChange={e => setSearchText(e.target.value)}
          style={{ width: 300 }}
          allowClear
        />
      </div>
      <Card>
        <Table dataSource={filteredData} columns={columns} rowKey="id" loading={loading} />
      </Card>
    </div>
  );
};

export default ConsultationListPage;
