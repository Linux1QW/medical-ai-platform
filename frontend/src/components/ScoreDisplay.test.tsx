import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import ScoreDisplay from './ScoreDisplay';

describe('ScoreDisplay', () => {
  it('默认 number 模式渲染分数与维度名', () => {
    render(<ScoreDisplay score={92} dimension="问诊技巧" />);
    expect(screen.getByText('92')).toBeInTheDocument();
    expect(screen.getByText('问诊技巧')).toBeInTheDocument();
  });

  it('number 模式 showLabel 时展示等级文案', () => {
    render(<ScoreDisplay score={92} showLabel />);
    expect(screen.getByText('优秀')).toBeInTheDocument();
  });

  it('分数颜色跟随评分色彩体系', () => {
    render(<ScoreDisplay score={55} />);
    // < 60 为红色
    expect(screen.getByText('55')).toHaveStyle({ color: 'rgb(255, 77, 79)' });
  });

  it('tag 模式渲染带维度前缀与等级后缀的标签', () => {
    const { container } = render(
      <ScoreDisplay score={75} dimension="沟通" mode="tag" showLabel />,
    );
    const tag = container.querySelector('.ant-tag');
    expect(tag).not.toBeNull();
    expect(tag!.textContent).toContain('沟通 75');
    expect(tag!.textContent).toContain('良好');
  });

  it('progress 模式渲染进度条', () => {
    const { container } = render(<ScoreDisplay score={68} mode="progress" showLabel />);
    expect(container.querySelector('.ant-progress')).not.toBeNull();
    expect(screen.getByText('及格')).toBeInTheDocument();
  });

  it('dashboard 模式渲染仪表盘进度圈', () => {
    const { container } = render(
      <ScoreDisplay score={88} mode="dashboard" dimension="总分" showLabel />,
    );
    expect(container.querySelector('.ant-progress-circle')).not.toBeNull();
    expect(screen.getByText('总分')).toBeInTheDocument();
    expect(screen.getByText('优秀')).toBeInTheDocument();
  });
});
