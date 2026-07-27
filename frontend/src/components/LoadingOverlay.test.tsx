import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import LoadingOverlay from './LoadingOverlay';

describe('LoadingOverlay', () => {
  it('loading=false 时直接渲染子元素，无遮罩', () => {
    const { container } = render(
      <LoadingOverlay loading={false}>
        <div>content-area</div>
      </LoadingOverlay>,
    );
    expect(screen.getByText('content-area')).toBeInTheDocument();
    expect(container.querySelector('.ant-spin')).toBeNull();
  });

  it('loading=true 时显示 Spin 遮罩与默认提示文字', () => {
    const { container } = render(
      <LoadingOverlay loading>
        <div>content-area</div>
      </LoadingOverlay>,
    );
    expect(container.querySelector('.ant-spin')).not.toBeNull();
    expect(screen.getByText('加载中...')).toBeInTheDocument();
    // 底层内容仍在（模糊化展示）
    expect(screen.getByText('content-area')).toBeInTheDocument();
  });

  it('支持自定义提示文字', () => {
    render(<LoadingOverlay loading tip="评估生成中" />);
    expect(screen.getByText('评估生成中')).toBeInTheDocument();
  });
});
