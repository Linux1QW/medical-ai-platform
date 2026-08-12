import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

// Mock livekit-client before importing VoiceConsultation
vi.mock('livekit-client', () => {
  const RoomEvent = {
    Disconnected: 'disconnected',
    Reconnecting: 'reconnecting',
    Reconnected: 'reconnected',
  };

  class MockRoom {
    private handlers: Record<string, ((...args: unknown[]) => void)[]> = {};
    connect = vi.fn().mockResolvedValue(undefined);
    disconnect = vi.fn();
    on(event: string, handler: (...args: unknown[]) => void) {
      if (!this.handlers[event]) this.handlers[event] = [];
      this.handlers[event].push(handler);
    }
  }

  return { Room: MockRoom, RoomEvent };
});

import { VoiceConsultation } from './VoiceConsultation';

describe('VoiceConsultation', () => {
  it('renders idle state with start button', () => {
    render(<VoiceConsultation consultationId={1} />);
    expect(screen.getByText('语音问诊 (Beta)')).toBeInTheDocument();
    expect(screen.getByText(/未连接/)).toBeInTheDocument();
    expect(screen.getByText('开始语音问诊')).toBeInTheDocument();
  });

  it('shows privacy notice about no raw audio', () => {
    render(<VoiceConsultation consultationId={1} />);
    expect(screen.getByText(/语音功能不会保存原始音频数据/)).toBeInTheDocument();
  });

  it('shows token validity notice', () => {
    render(<VoiceConsultation consultationId={1} />);
    expect(screen.getByText(/Token 有效期: 10 分钟/)).toBeInTheDocument();
  });
});
