import React from 'react';
import { Spin } from 'antd';

export interface LoadingOverlayProps {
  /** 是否显示加载状态 */
  loading: boolean;
  /** 加载提示文字 */
  tip?: string;
  /** 子元素 */
  children?: React.ReactNode;
}

const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  loading,
  tip = '加载中...',
  children,
}) => {
  if (!loading) return <>{children}</>;

  return (
    <div style={{ position: 'relative', width: '100%', minHeight: 200 }}>
      <div
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'rgba(255,255,255,0.75)',
          zIndex: 10,
          borderRadius: 8,
        }}
      >
        <Spin size="large" />
        <div style={{ marginTop: 12, color: '#666', fontSize: 14 }}>{tip}</div>
      </div>
      {/* 底层内容（模糊化） */}
      <div style={{ filter: 'blur(2px)', pointerEvents: 'none', userSelect: 'none' }}>
        {children}
      </div>
    </div>
  );
};

export default LoadingOverlay;
