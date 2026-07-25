import { describe, expect, it } from 'vitest';

import { getScoreAntTagColor, getScoreColor, getScoreLevel } from './score';

describe('score 工具函数', () => {
  it('getScoreColor 按分数段返回颜色', () => {
    expect(getScoreColor(90)).toBe('#52c41a');
    expect(getScoreColor(85)).toBe('#52c41a');
    expect(getScoreColor(75)).toBe('#1890ff');
    expect(getScoreColor(65)).toBe('#faad14');
    expect(getScoreColor(59)).toBe('#ff4d4f');
  });

  it('getScoreAntTagColor 与色彩体系一致', () => {
    expect(getScoreAntTagColor(85)).toBe('success');
    expect(getScoreAntTagColor(70)).toBe('processing');
    expect(getScoreAntTagColor(60)).toBe('warning');
    expect(getScoreAntTagColor(0)).toBe('error');
  });

  it('getScoreLevel 返回等级文案与对应颜色', () => {
    expect(getScoreLevel(92)).toEqual({ text: '优秀', color: '#52c41a' });
    expect(getScoreLevel(70)).toEqual({ text: '良好', color: '#1890ff' });
    expect(getScoreLevel(60)).toEqual({ text: '及格', color: '#faad14' });
    expect(getScoreLevel(30)).toEqual({ text: '待提升', color: '#ff4d4f' });
  });
});
