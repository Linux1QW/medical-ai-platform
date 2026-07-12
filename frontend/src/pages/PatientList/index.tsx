import React, { useEffect, useState, useCallback } from 'react';
import { Card, Table, Select, Button, Space, Typography, Rate, message } from 'antd';
import { PlayCircleOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { getPatients } from '../../api/patient';
import { startConsultation } from '../../api/consultation';
import { PersonalityTag } from '../../components';
import type { VirtualPatient } from '../../types';

const { Title } = Typography;

const PatientListPage: React.FC = () => {
  const navigate = useNavigate();
  const [patients, setPatients] = useState<VirtualPatient[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string | undefined>();

  const fetchPatients = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getPatients(filter ? { personality_type: filter } : undefined);
      setPatients(data);
    } catch {
      message.error('加载患者列表失败');
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => { fetchPatients(); }, [fetchPatients]);

  const handleStart = async (patientId: number) => {
    try {
      const consultation = await startConsultation(patientId);
      message.success('问诊已创建');
      navigate(`/consultation/${consultation.id}`);
    } catch {
      // 错误已由请求拦截器处理并显示
    }
  };

  const columns = [
    { title: '姓名', dataIndex: 'name', key: 'name' },
    { title: '年龄', dataIndex: 'age', key: 'age', width: 80 },
    {
      title: '性别', dataIndex: 'gender', key: 'gender', width: 80,
      render: (v: string) => (v === 'male' ? '男' : '女'),
    },
    {
      title: '人格类型', dataIndex: 'personality_type', key: 'personality_type',
      render: (v: string) => <PersonalityTag type={v} />,
    },
    { title: '主诉', dataIndex: 'chief_complaint', key: 'chief_complaint', ellipsis: true },
    {
      title: '难度', dataIndex: 'difficulty_level', key: 'difficulty_level', width: 140,
      render: (v: number) => <Rate disabled value={v} count={5} />,
    },
    {
      title: '操作', key: 'action', width: 120,
      render: (_: unknown, record: VirtualPatient) => (
        <Button type="primary" icon={<PlayCircleOutlined />} onClick={() => handleStart(record.id)}>
          开始问诊
        </Button>
      ),
    },
  ];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>虚拟患者列表</Title>
        <Space>
          <span>人格类型筛选：</span>
          <Select
            allowClear
            placeholder="全部"
            style={{ width: 140 }}
            value={filter}
            onChange={setFilter}
            options={[
              { value: '配合型', label: '配合型' },
              { value: '焦虑型', label: '焦虑型' },
              { value: '沉默型', label: '沉默型' },
              { value: '对抗型', label: '对抗型' },
            ]}
          />
        </Space>
      </div>
      <Card>
        <Table dataSource={patients} columns={columns} rowKey="id" loading={loading} />
      </Card>
    </div>
  );
};

export default PatientListPage;
