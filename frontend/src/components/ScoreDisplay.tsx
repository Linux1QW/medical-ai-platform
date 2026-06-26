import React from 'react';
import { Progress, Tag, Typography } from 'antd';

const { Text } = Typography;

/**
 * 统一评分色彩体系
 * >= 85: 绿色  |  >= 70: 蓝色  |  >= 60: 橙色  |  < 60: 红色
 */
export const getScoreColor = (score: number): string => {
  if (score >= 85) return '#52c41a';
  if (score >= 70) return '#1890ff';
  if (score >= 60) return '#faad14';
  return '#ff4d4f';
};

export const getScoreLevel = (score: number): { text: string; color: string } => {
  const color = getScoreColor(score);
  if (score >= 85) return { text: '优秀', color };
  if (score >= 70) return { text: '良好', color };
  if (score >= 60) return { text: '及格', color };
  return { text: '待提升', color };
};

export interface ScoreDisplayProps {
  /** 分数 0-100 */
  score: number;
  /** 维度名称 */
  dimension?: string;
  /** 是否显示文字标签（优秀/良好/及格/待提升） */
  showLabel?: boolean;
  /** 展示模式：number=纯数字, progress=进度条, tag=标签, dashboard=仪表盘 */
  size?: 'small' | 'default' | 'large';
  /** 展示模式 */
  mode?: 'number' | 'progress' | 'tag' | 'dashboard';
}

const fontSizeMap = { small: 18, default: 28, large: 42 };
const tagFontSizeMap = { small: 11, default: 12, large: 14 };

const ScoreDisplay: React.FC<ScoreDisplayProps> = ({
  score,
  dimension,
  showLabel = false,
  size = 'default',
  mode = 'number',
}) => {
  const color = getScoreColor(score);
  const level = getScoreLevel(score);

  if (mode === 'tag') {
    return (
      <Tag color={color} style={{ borderRadius: 10, fontSize: size === 'large' ? 16 : undefined }}>
        {dimension ? `${dimension} ` : ''}{score}
        {showLabel ? ` ${level.text}` : ''}
      </Tag>
    );
  }

  if (mode === 'progress') {
    return (
      <div>
        {dimension && <Text type="secondary" style={{ fontSize: 12 }}>{dimension}</Text>}
        <Progress
          percent={score}
          strokeColor={color}
          size={size === 'small' ? 'small' : 'default'}
          format={() => <span style={{ color }}>{score}</span>}
        />
        {showLabel && <Tag color={color} style={{ borderRadius: 10 }}>{level.text}</Tag>}
      </div>
    );
  }

  if (mode === 'dashboard') {
    return (
      <div style={{ textAlign: 'center' }}>
        <Progress
          type="dashboard"
          percent={score}
          size={size === 'small' ? 80 : size === 'large' ? 200 : 140}
          strokeColor={color}
          format={(p) => <span style={{ fontSize: size === 'large' ? 42 : 28, fontWeight: 700 }}>{p}</span>}
        />
        {dimension && <div style={{ marginTop: 8, fontSize: size === 'large' ? 18 : 14, fontWeight: 600 }}>{dimension}</div>}
        {showLabel && <Tag color={color} style={{ borderRadius: 10, marginTop: 4 }}>{level.text}</Tag>}
      </div>
    );
  }

  // default: number mode
  return (
    <div style={{ textAlign: 'center' }}>
      {dimension && <div style={{ fontSize: 13, color: '#888', marginBottom: 4 }}>{dimension}</div>}
      <div style={{ fontSize: fontSizeMap[size], fontWeight: 700, color }}>
        {score}
      </div>
      {showLabel && (
        <Tag color={color} style={{ borderRadius: 10, marginTop: 4, fontSize: tagFontSizeMap[size] }}>
          {level.text}
        </Tag>
      )}
    </div>
  );
};

export default ScoreDisplay;
