import React from 'react';
import { Layout, Menu } from 'antd';
import {
  HomeOutlined,
  MessageOutlined,
  TeamOutlined,
  BarChartOutlined,
  SettingOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { useNavigate, useLocation } from 'react-router-dom';

const { Sider } = Layout;

interface SidebarProps {
  collapsed?: boolean;
  isAdmin?: boolean;
}

const menuItems = [
  { key: '/dashboard', icon: <HomeOutlined />, label: '工作台' },
  { key: '/patients', icon: <TeamOutlined />, label: '虚拟患者' },
  { key: '/consultations', icon: <MessageOutlined />, label: '我的问诊' },
  { key: '/stats', icon: <BarChartOutlined />, label: '数据统计' },
];

const adminMenuItems = [
  ...menuItems,
  { type: 'divider' as const },
  { key: '/admin/consultations', icon: <MessageOutlined />, label: '全部问诊' },
  { key: '/admin/patients', icon: <SettingOutlined />, label: '患者管理' },
  { key: '/profile', icon: <UserOutlined />, label: '个人资料' },
];

const Sidebar: React.FC<SidebarProps> = ({ collapsed = false, isAdmin = false }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const items = isAdmin ? adminMenuItems : menuItems;

  return (
    <Sider
      width={220}
      collapsed={collapsed}
      theme="light"
      style={{ borderRight: '1px solid #f0f0f0' }}
    >
      <div
        style={{
          height: 64,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontWeight: 700,
          fontSize: collapsed ? 14 : 16,
          color: '#4f46e5',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
        }}
      >
        <MessageOutlined style={{ fontSize: 20, marginRight: collapsed ? 0 : 8 }} />
        {!collapsed && <span>临床问诊评估</span>}
      </div>
      <Menu
        mode="inline"
        selectedKeys={[location.pathname]}
        items={items}
        onClick={({ key }) => navigate(key)}
        style={{ borderRight: 0 }}
      />
    </Sider>
  );
};

export default Sidebar;
