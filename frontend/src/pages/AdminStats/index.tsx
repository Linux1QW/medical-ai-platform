import React, { useEffect, useState } from 'react';
import { Card, Col, Row, Statistic, Typography, Spin, Table, Tag } from 'antd';
import {
  BarChartOutlined,
  MessageOutlined,
  FileTextOutlined,
} from '@ant-design/icons';
import { getStats } from '../../api/evaluation';
import { useAuth } from '../../store/useAuth';
import type { StatsSummary, UserStatItem } from '../../types';

const { Title } = Typography;

const AdminStatsPage: React.FC = () => {
  const { isAdmin } = useAuth();
  const [stats, setStats] = useState<StatsSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getStats()
      .then(setStats)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Spin style={{ display: 'block', margin: '100px auto' }} size="large" />;
  if (!stats) return <div>加载失败</div>;

  const dimensionData = [
    { key: '1', dimension: '病史采集', avg: stats.avg_inquiry_score },
    { key: '2', dimension: '医学知识', avg: stats.avg_knowledge_score },
    { key: '3', dimension: '沟通交流', avg: stats.avg_humanistic_score },
    { key: '4', dimension: '诊断结果', avg: stats.avg_diagnosis_score },
    { key: '5', dimension: '治疗方案', avg: stats.avg_treatment_score },
  ];

  const tagColor = (score: number) => {
    if (score >= 80) return 'success';
    if (score >= 60) return 'warning';
    return 'error';
  };

  const isPlatformStats = isAdmin;
  return (
    <div>
      <Title level={4}>{isPlatformStats ? '平台数据统计' : '我的数据统计'}</Title>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card><Statistic title={isPlatformStats ? '总实训次数' : '我的实训次数'} value={stats.total_consultations} prefix={<MessageOutlined />} /></Card>
        </Col>
        <Col span={8}>
          <Card><Statistic title={isPlatformStats ? '总评估报告数' : '我的评估报告数'} value={stats.total_evaluations} prefix={<FileTextOutlined />} /></Card>
        </Col>
        <Col span={8}>
          <Card><Statistic title="平均综合评分" value={stats.avg_total_score} prefix={<BarChartOutlined />} suffix="/ 100" precision={1} /></Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={14}>
          <Card title="各维度平均评分">
            <Table
              dataSource={dimensionData}
              pagination={false}
              size="small"
              columns={[
                { title: '评估维度', dataIndex: 'dimension', key: 'dimension' },
                {
                  title: '平均分',
                  dataIndex: 'avg',
                  key: 'avg',
                  render: (v: number) => (
                    <span>
                      <Tag color={tagColor(v)}>{v.toFixed(1)}</Tag>
                      <span style={{ color: '#999', fontSize: 12 }}>/ 100</span>
                    </span>
                  ),
                  sorter: (a: { avg: number }, b: { avg: number }) => a.avg - b.avg,
                },
                {
                  title: '水平',
                  dataIndex: 'avg',
                  key: 'level',
                  render: (v: number) => {
                    if (v >= 90) return <Tag color="success">优秀</Tag>;
                    if (v >= 80) return <Tag color="processing">良好</Tag>;
                    if (v >= 60) return <Tag color="warning">一般</Tag>;
                    return <Tag color="error">不及格</Tag>;
                  },
                },
              ]}
            />
          </Card>
        </Col>
        <Col span={10}>
          <Card title="综合评分分布（按用户平均分）">
            {stats.score_distribution.length > 0 ? (
              <Table
                dataSource={stats.score_distribution.map((d, i) => ({ key: i, ...d }))}
                pagination={false}
                size="small"
                columns={[
                  { title: '评分区间', dataIndex: 'range', key: 'range' },
                  {
                    title: '用户数',
                    dataIndex: 'count',
                    key: 'count',
                    render: (v: number) => <Tag color={v > 0 ? 'blue' : 'default'}>{v}</Tag>,
                  },
                ]}
              />
            ) : (
              <div style={{ textAlign: 'center', padding: 40, color: '#999' }}>暂无数据</div>
            )}
          </Card>
        </Col>
      </Row>

      {isAdmin && stats.user_stats && stats.user_stats.length > 0 && (
        <Card title="各用户统计明细" style={{ marginBottom: 24 }}>
          <Table<UserStatItem>
            dataSource={stats.user_stats}
            rowKey="user_id"
            size="middle"
            pagination={{ pageSize: 10 }}
            scroll={{ x: 1200 }}
            columns={[
              {
                title: '用户名',
                dataIndex: 'username',
                key: 'username',
                fixed: 'left' as const,
                width: 100,
              },
              {
                title: '姓名',
                dataIndex: 'real_name',
                key: 'real_name',
                width: 100,
                render: (v: string) => v || '-',
              },
              {
                title: '科室',
                dataIndex: 'department',
                key: 'department',
                width: 100,
                render: (v: string) => v ? <Tag>{v}</Tag> : '-',
              },
              {
                title: '实训次数',
                dataIndex: 'total_consultations',
                key: 'total_consultations',
                width: 90,
                sorter: (a: UserStatItem, b: UserStatItem) => a.total_consultations - b.total_consultations,
              },
              {
                title: '评估次数',
                dataIndex: 'total_evaluations',
                key: 'total_evaluations',
                width: 90,
                sorter: (a: UserStatItem, b: UserStatItem) => a.total_evaluations - b.total_evaluations,
              },
              {
                title: '病史采集',
                dataIndex: 'avg_inquiry_score',
                key: 'avg_inquiry_score',
                width: 90,
                sorter: (a: UserStatItem, b: UserStatItem) => a.avg_inquiry_score - b.avg_inquiry_score,
                render: (v: number) => <Tag color={tagColor(v)}>{v.toFixed(1)}</Tag>,
              },
              {
                title: '医学知识',
                dataIndex: 'avg_knowledge_score',
                key: 'avg_knowledge_score',
                width: 90,
                sorter: (a: UserStatItem, b: UserStatItem) => a.avg_knowledge_score - b.avg_knowledge_score,
                render: (v: number) => <Tag color={tagColor(v)}>{v.toFixed(1)}</Tag>,
              },
              {
                title: '沟通交流',
                dataIndex: 'avg_humanistic_score',
                key: 'avg_humanistic_score',
                width: 90,
                sorter: (a: UserStatItem, b: UserStatItem) => a.avg_humanistic_score - b.avg_humanistic_score,
                render: (v: number) => <Tag color={tagColor(v)}>{v.toFixed(1)}</Tag>,
              },
              {
                title: '诊断结果',
                dataIndex: 'avg_diagnosis_score',
                key: 'avg_diagnosis_score',
                width: 90,
                sorter: (a: UserStatItem, b: UserStatItem) => a.avg_diagnosis_score - b.avg_diagnosis_score,
                render: (v: number) => <Tag color={tagColor(v)}>{v.toFixed(1)}</Tag>,
              },
              {
                title: '治疗方案',
                dataIndex: 'avg_treatment_score',
                key: 'avg_treatment_score',
                width: 90,
                sorter: (a: UserStatItem, b: UserStatItem) => a.avg_treatment_score - b.avg_treatment_score,
                render: (v: number) => <Tag color={tagColor(v)}>{v.toFixed(1)}</Tag>,
              },
              {
                title: '综合评分',
                dataIndex: 'avg_total_score',
                key: 'avg_total_score',
                width: 90,
                defaultSortOrder: 'descend' as const,
                sorter: (a: UserStatItem, b: UserStatItem) => a.avg_total_score - b.avg_total_score,
                render: (v: number) => <Tag color={tagColor(v)} style={{ fontWeight: 600 }}>{v.toFixed(1)}</Tag>,
              },
            ]}
          />
        </Card>
      )}
    </div>
  );
};

export default AdminStatsPage;
