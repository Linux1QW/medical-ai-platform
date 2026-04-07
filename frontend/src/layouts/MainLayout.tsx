import React from 'react';
import { Layout, Menu, Button, Avatar, Dropdown, theme } from 'antd';
import {
  UserOutlined,
  MessageOutlined,
  BarChartOutlined,
  TeamOutlined,
  LogoutOutlined,
  HomeOutlined,
  EditOutlined,
} from '@ant-design/icons';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../store/useAuth';

const { Header, Sider, Content } = Layout;

const MainLayout: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, isAdmin, logout } = useAuth();
  const { token: themeToken } = theme.useToken();

  const menuItems = [
    { key: '/dashboard', icon: <HomeOutlined />, label: '工作台' },
    { key: '/patients', icon: <TeamOutlined />, label: '虚拟患者' },
    { key: '/consultations', icon: <MessageOutlined />, label: '我的问诊' },
    { key: '/stats', icon: <BarChartOutlined />, label: '数据统计' },
    ...(isAdmin ? [
      { key: '/admin/consultations', icon: <MessageOutlined />, label: '全部问诊' },
      { key: '/admin/patients', icon: <EditOutlined />, label: '患者管理' }
    ] : []),
  ];

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const dropdownItems = {
    items: [
      { key: 'profile', icon: <UserOutlined />, label: '个人资料', onClick: () => navigate('/profile') },
      { key: 'logout', icon: <LogoutOutlined />, label: '退出登录', onClick: handleLogout },
    ],
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider width={220} theme="light" style={{ borderRight: `1px solid ${themeToken.colorBorderSecondary}` }}>
        <div style={{ height: 64, display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: 16, color: themeToken.colorPrimary }}>
          临床问诊评估平台
        </div>
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          style={{ borderRight: 0 }}
        />
      </Sider>
      <Layout>
        <Header style={{ background: '#fff', padding: '0 24px', display: 'flex', justifyContent: 'flex-end', alignItems: 'center', borderBottom: `1px solid ${themeToken.colorBorderSecondary}` }}>
          <Dropdown menu={dropdownItems}>
            <Button type="text" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Avatar size="small" icon={<UserOutlined />} />
              <span>{user?.real_name || user?.username}</span>
            </Button>
          </Dropdown>
        </Header>
        <Content style={{ margin: 24, padding: 24, background: '#fff', borderRadius: 8, overflow: 'auto' }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
};

export default MainLayout;
