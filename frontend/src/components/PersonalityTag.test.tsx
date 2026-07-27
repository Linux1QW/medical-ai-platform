import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import PersonalityTag from './PersonalityTag';

describe('PersonalityTag', () => {
  it.each([
    ['配合型', 'green'],
    ['焦虑型', 'orange'],
    ['沉默型', 'blue'],
    ['对抗型', 'red'],
  ])('已知类型 %s 渲染 %s 色标签', (type, color) => {
    const { container } = render(<PersonalityTag type={type} />);
    expect(screen.getByText(type)).toBeInTheDocument();
    expect(container.querySelector(`.ant-tag-${color}`)).not.toBeNull();
  });

  it('未知类型回退为无色默认标签', () => {
    const { container } = render(<PersonalityTag type="神秘型" />);
    expect(screen.getByText('神秘型')).toBeInTheDocument();
    expect(
      container.querySelector('.ant-tag-green, .ant-tag-orange, .ant-tag-blue, .ant-tag-red'),
    ).toBeNull();
  });

  it('showIcon 时渲染类型对应图标', () => {
    const { container } = render(<PersonalityTag type="配合型" showIcon />);
    expect(container.querySelector('.anticon-smile')).not.toBeNull();
  });

  it('默认不渲染图标', () => {
    const { container } = render(<PersonalityTag type="配合型" />);
    expect(container.querySelector('.anticon')).toBeNull();
  });
});
