import React from 'react';
import { Tag } from 'antd';
import {
  SmileOutlined,
  WarningOutlined,
  MehOutlined,
  FrownOutlined,
} from '@ant-design/icons';

export type PersonalityType = '配合型' | '焦虑型' | '沉默型' | '对抗型';

const config: Record<PersonalityType, { color: string; icon: React.ReactNode }> = {
  配合型: { color: 'green', icon: <SmileOutlined /> },
  焦虑型: { color: 'orange', icon: <WarningOutlined /> },
  沉默型: { color: 'blue', icon: <MehOutlined /> },
  对抗型: { color: 'red', icon: <FrownOutlined /> },
};

export interface PersonalityTagProps {
  /** 人格类型 */
  type: PersonalityType | string;
  /** 是否显示图标 */
  showIcon?: boolean;
}

const PersonalityTag: React.FC<PersonalityTagProps> = ({ type, showIcon = false }) => {
  const cfg = config[type as PersonalityType];
  if (!cfg) return <Tag>{type}</Tag>;
  return (
    <Tag color={cfg.color} icon={showIcon ? cfg.icon : undefined}>
      {type}
    </Tag>
  );
};

export default PersonalityTag;
