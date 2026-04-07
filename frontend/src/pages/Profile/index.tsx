import React, { useEffect } from 'react';
import { Card, Form, Input, Button, Typography, message } from 'antd';
import { useAuth } from '../../store/useAuth';
import { updateProfile } from '../../api/auth';

const { Title } = Typography;

const ProfilePage: React.FC = () => {
  const { user, token, saveAuth } = useAuth();
  const [form] = Form.useForm();

  useEffect(() => {
    if (user) {
      form.setFieldsValue({
        username: user.username,
        email: user.email,
        real_name: user.real_name,
        department: user.department,
      });
    }
  }, [user, form]);

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      const updated = await updateProfile({
        email: values.email,
        real_name: values.real_name,
        department: values.department,
      });
      if (user && updated && token) {
        saveAuth(token, { ...user, ...updated });
      }
      message.success('个人资料更新成功');
    } catch (err) {
      if (err instanceof Error && err.message) {
        message.error(err.message);
      }
    }
  };

  if (!user) return null;

  return (
    <div>
      <Title level={4}>个人资料</Title>
      <Card style={{ maxWidth: 480 }}>
        <Form form={form} layout="vertical" onFinish={handleSubmit}>
          <Form.Item name="username" label="用户名">
            <Input disabled placeholder="用户名" />
          </Form.Item>
          <Form.Item
            name="email"
            label="邮箱"
            rules={[{ required: true, message: '请输入邮箱' }, { type: 'email', message: '请输入有效邮箱' }]}
          >
            <Input placeholder="邮箱" />
          </Form.Item>
          <Form.Item name="real_name" label="真实姓名">
            <Input placeholder="真实姓名" />
          </Form.Item>
          <Form.Item name="department" label="科室">
            <Input placeholder="科室" />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit">
              保存
            </Button>
          </Form.Item>
        </Form>
      </Card>
    </div>
  );
};

export default ProfilePage;
