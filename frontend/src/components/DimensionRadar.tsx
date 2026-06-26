import React from 'react';
import { Typography } from 'antd';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';

const { Title } = Typography;

export interface DimensionScores {
  病史采集?: number | null;
  医学知识?: number | null;
  人文关怀?: number | null;
  诊断结果?: number | null;
  治疗方案?: number | null;
  [key: string]: number | null | undefined;
}

export interface DimensionRadarProps {
  /** 五维分数对象 */
  scores: DimensionScores;
  /** 图表标题 */
  title?: string;
  /** 图表高度(px) */
  height?: number;
}

const DimensionRadar: React.FC<DimensionRadarProps> = ({
  scores,
  title,
  height = 320,
}) => {
  const data = Object.entries(scores)
    .filter(([, v]) => v != null)
    .map(([subject, score]) => ({
      subject,
      score: score as number,
      fullMark: 100,
    }));

  if (data.length === 0) return null;

  return (
    <div>
      {title && (
        <Title level={5} style={{ textAlign: 'center', marginBottom: 8 }}>
          {title}
        </Title>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
          <PolarGrid />
          <PolarAngleAxis dataKey="subject" />
          <PolarRadiusAxis angle={30} domain={[0, 100]} />
          <Tooltip />
          <Radar
            name="评估得分"
            dataKey="score"
            stroke="#1677ff"
            fill="#1677ff"
            fillOpacity={0.6}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default DimensionRadar;
