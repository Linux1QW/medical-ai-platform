/** 统一评分色彩体系：>= 85 绿 | >= 70 蓝 | >= 60 橙 | < 60 红 */

export const getScoreColor = (score: number): string => {
  if (score >= 85) return '#52c41a';
  if (score >= 70) return '#1890ff';
  if (score >= 60) return '#faad14';
  return '#ff4d4f';
};

export const getScoreAntTagColor = (score: number): 'success' | 'processing' | 'warning' | 'error' => {
  if (score >= 85) return 'success';
  if (score >= 70) return 'processing';
  if (score >= 60) return 'warning';
  return 'error';
};

export const getScoreLevel = (score: number): { text: string; color: string } => {
  const color = getScoreColor(score);
  if (score >= 85) return { text: '优秀', color };
  if (score >= 70) return { text: '良好', color };
  if (score >= 60) return { text: '及格', color };
  return { text: '待提升', color };
};
