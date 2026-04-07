import React, { useEffect, useState } from 'react';
import {
  Card,
  Table,
  Button,
  Modal,
  Form,
  Input,
  Select,
  InputNumber,
  Popconfirm,
  message,
  Typography,
  Space,
  Tag,
  Rate,
  Spin,
} from 'antd';
import { PlusOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons';
import {
  getPatients,
  getPatient,
  createPatient,
  updatePatient,
  deletePatient,
} from '../../api/patient';
import type { VirtualPatient } from '../../types';

const { Title } = Typography;
const { TextArea } = Input;

const personalityMap: Record<string, { color: string; text: string }> = {
  配合型: { color: 'green', text: '配合型' },
  焦虑型: { color: 'orange', text: '焦虑型' },
  沉默型: { color: 'blue', text: '沉默型' },
  对抗型: { color: 'red', text: '对抗型' },
};

type PatientFormValues = {
  name: string;
  age: number;
  gender: 'male' | 'female';
  personality_type: '配合型' | '焦虑型' | '沉默型' | '对抗型';
  chief_complaint: string;
  medical_history: string;
  symptoms: string;
  expected_diagnosis: string;
  system_prompt: string;
  difficulty_level: number;
};

const AdminPatientsPage: React.FC = () => {
  const [patients, setPatients] = useState<VirtualPatient[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [editingPatient, setEditingPatient] = useState<VirtualPatient | null>(null);
  const [formLoading, setFormLoading] = useState(false);
  const [formDataToSet, setFormDataToSet] = useState<PatientFormValues | null>(null);
  const [form] = Form.useForm<PatientFormValues>();

  useEffect(() => {
    if (modalVisible && formDataToSet && !formLoading) {
      form.setFieldsValue(formDataToSet);
      setFormDataToSet(null);
    }
  }, [modalVisible, formDataToSet, formLoading, form]);

  const fetchPatients = async () => {
    setLoading(true);
    try {
      const data = await getPatients();
      setPatients(data);
    } catch {
      message.error('加载患者列表失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPatients();
  }, []);

  const handleAdd = () => {
    setEditingPatient(null);
    form.resetFields();
    setModalVisible(true);
  };

  const handleEdit = async (record: VirtualPatient) => {
    form.resetFields();
    setEditingPatient(null);
    setFormDataToSet(null);
    setModalVisible(true);
    setFormLoading(true);
    try {
      const full = await getPatient(record.id);
      setEditingPatient(full);
      setFormDataToSet({
        name: full.name,
        age: full.age,
        gender: full.gender,
        personality_type: full.personality_type,
        chief_complaint: full.chief_complaint,
        medical_history: full.medical_history ?? '',
        symptoms: full.symptoms ?? '',
        expected_diagnosis: full.expected_diagnosis ?? '',
        system_prompt: full.system_prompt ?? '',
        difficulty_level: full.difficulty_level,
      });
    } catch {
      message.error('加载患者详情失败');
      setModalVisible(false);
    } finally {
      setFormLoading(false);
    }
  };

  const handleModalOk = async () => {
    try {
      const values = await form.validateFields();
      const payload = {
        name: values.name,
        age: values.age,
        gender: values.gender,
        personality_type: values.personality_type,
        chief_complaint: values.chief_complaint,
        medical_history: values.medical_history,
        symptoms: values.symptoms,
        expected_diagnosis: values.expected_diagnosis,
        system_prompt: values.system_prompt,
        difficulty_level: values.difficulty_level,
      };
      if (editingPatient) {
        await updatePatient(editingPatient.id, payload);
        message.success('更新成功');
      } else {
        await createPatient(payload);
        message.success('添加成功');
      }
      setModalVisible(false);
      fetchPatients();
    } catch (err) {
      if (err instanceof Error && err.message) {
        message.error(err.message);
      }
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await deletePatient(id);
      message.success('删除成功');
      fetchPatients();
    } catch {
      message.error('删除失败');
    }
  };

  const columns = [
    { title: '姓名', dataIndex: 'name', key: 'name', width: 100 },
    { title: '年龄', dataIndex: 'age', key: 'age', width: 70 },
    {
      title: '性别',
      dataIndex: 'gender',
      key: 'gender',
      width: 70,
      render: (v: string) => (v === 'male' ? '男' : '女'),
    },
    {
      title: '人格类型',
      dataIndex: 'personality_type',
      key: 'personality_type',
      width: 100,
      render: (v: string) => <Tag color={personalityMap[v]?.color}>{personalityMap[v]?.text}</Tag>,
    },
    {
      title: '主诉',
      dataIndex: 'chief_complaint',
      key: 'chief_complaint',
      ellipsis: true,
    },
    {
      title: '难度',
      dataIndex: 'difficulty_level',
      key: 'difficulty_level',
      width: 120,
      render: (v: number) => <Rate disabled value={v} count={5} />,
    },
    {
      title: '操作',
      key: 'action',
      width: 160,
      render: (_: unknown, record: VirtualPatient) => (
        <Space>
          <Button type="link" size="small" icon={<EditOutlined />} onClick={() => handleEdit(record)}>
            编辑
          </Button>
          <Popconfirm
            title="确定删除该患者？"
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
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>
          虚拟患者管理
        </Title>
        <Button type="primary" icon={<PlusOutlined />} onClick={handleAdd}>
          添加患者
        </Button>
      </div>
      <Card>
        <Table
          dataSource={patients}
          columns={columns}
          rowKey="id"
          loading={loading}
        />
      </Card>

      <Modal
        title={editingPatient ? '编辑患者' : '添加患者'}
        open={modalVisible}
        onOk={handleModalOk}
        onCancel={() => {
          setModalVisible(false);
          setEditingPatient(null);
        }}
        width={600}
        destroyOnClose
        confirmLoading={formLoading}
      >
        <Form form={form} layout="vertical" preserve={false} style={{ position: 'relative' }}>
          {formLoading && (
            <div
              style={{
                position: 'absolute',
                inset: 0,
                background: 'rgba(255,255,255,0.8)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 10,
              }}
            >
              <Spin tip="加载患者信息中..." />
            </div>
          )}
          <Form.Item
            name="name"
            label="姓名"
            rules={[{ required: true, message: '请输入姓名' }]}
          >
            <Input placeholder="患者姓名" />
          </Form.Item>
          <Form.Item
            name="age"
            label="年龄"
            rules={[{ required: true, message: '请输入年龄' }]}
          >
            <InputNumber min={1} max={150} style={{ width: '100%' }} placeholder="年龄" />
          </Form.Item>
          <Form.Item
            name="gender"
            label="性别"
            rules={[{ required: true, message: '请选择性别' }]}
          >
            <Select
              placeholder="选择性别"
              options={[
                { value: 'male', label: '男' },
                { value: 'female', label: '女' },
              ]}
            />
          </Form.Item>
          <Form.Item
            name="personality_type"
            label="人格类型"
            rules={[{ required: true, message: '请选择人格类型' }]}
          >
            <Select
              placeholder="选择人格类型"
              options={[
                { value: '配合型', label: '配合型' },
                { value: '焦虑型', label: '焦虑型' },
                { value: '沉默型', label: '沉默型' },
                { value: '对抗型', label: '对抗型' },
              ]}
            />
          </Form.Item>
          <Form.Item
            name="chief_complaint"
            label="主诉"
            rules={[{ required: true, message: '请输入主诉' }]}
          >
            <Input placeholder="主诉" />
          </Form.Item>
          <Form.Item name="medical_history" label="病史">
            <TextArea rows={3} placeholder="病史" />
          </Form.Item>
          <Form.Item name="symptoms" label="症状">
            <TextArea rows={3} placeholder="症状" />
          </Form.Item>
          <Form.Item name="expected_diagnosis" label="预期诊断">
            <Input placeholder="预期诊断" />
          </Form.Item>
          <Form.Item name="system_prompt" label="系统提示词">
            <TextArea rows={4} placeholder="虚拟患者系统提示词" />
          </Form.Item>
          <Form.Item
            name="difficulty_level"
            label="难度等级"
            rules={[{ required: true, message: '请选择难度' }]}
          >
            <Select
              placeholder="选择难度 (1-5)"
              options={[
                { value: 1, label: '1 星' },
                { value: 2, label: '2 星' },
                { value: 3, label: '3 星' },
                { value: 4, label: '4 星' },
                { value: 5, label: '5 星' },
              ]}
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default AdminPatientsPage;
