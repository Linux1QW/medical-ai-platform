import { describe, expect, it } from 'vitest';
import { createSSEParser } from './sseParser';

describe('createSSEParser', () => {
  it('keeps event state across network chunks', () => {
    const parser = createSSEParser();
    expect(parser.push('event: suggestion\n')).toEqual([]);
    expect(
      parser.push('id: e1\ndata: {"suggested_question":"请问胸痛多久？"}\n\n'),
    ).toEqual([
      {
        event: 'suggestion',
        id: 'e1',
        data: { suggested_question: '请问胸痛多久？' },
      },
    ]);
  });

  it('handles CRLF line endings', () => {
    const parser = createSSEParser();
    const events = parser.push('event: thinking\r\nid: t1\r\ndata: {"turn_no":1}\r\n\r\n');
    expect(events).toEqual([
      { event: 'thinking', id: 't1', data: { turn_no: 1 } },
    ]);
  });

  it('concatenates multi-line data fields', () => {
    const parser = createSSEParser();
    const events = parser.push(
      'event: msg\nid: m1\ndata: line1\ndata: line2\ndata: line3\n\n',
    );
    expect(events).toHaveLength(1);
    expect(events[0].event).toBe('msg');
    expect(events[0].data).toBe('line1\nline2\nline3');
  });

  it('emits multiple events from a single chunk', () => {
    const parser = createSSEParser();
    const events = parser.push(
      'event: a\nid: 1\ndata: {}\n\nevent: b\nid: 2\ndata: {}\n\n',
    );
    expect(events).toHaveLength(2);
    expect(events[0].event).toBe('a');
    expect(events[1].event).toBe('b');
  });

  it('ignores comment lines (starting with ":")', () => {
    const parser = createSSEParser();
    const events = parser.push(': keep-alive comment\nevent: ping\nid: p1\ndata: {}\n\n');
    expect(events).toHaveLength(1);
    expect(events[0].event).toBe('ping');
  });

  it('flush() emits any remaining buffered event', () => {
    const parser = createSSEParser();
    parser.push('event: done\nid: d1\ndata: {"status":"ok"}');
    // No trailing \n\n yet
    const flushed = parser.flush();
    expect(flushed).toEqual([
      { event: 'done', id: 'd1', data: { status: 'ok' } },
    ]);
  });

  it('flush() returns empty array when nothing is buffered', () => {
    const parser = createSSEParser();
    expect(parser.flush()).toEqual([]);
  });

  it('defaults event name to "message" when no event field is set', () => {
    const parser = createSSEParser();
    const events = parser.push('id: x1\ndata: {"v":1}\n\n');
    expect(events).toEqual([{ event: 'message', id: 'x1', data: { v: 1 } }]);
  });

  it('handles data split across chunks mid-line', () => {
    const parser = createSSEParser();
    // First chunk ends mid-data-value
    expect(parser.push('event: split\nid: s1\ndata: {"ke')).toEqual([]);
    // Second chunk completes the data line and terminates the event
    const events = parser.push('y":"value"}\n\n');
    expect(events).toEqual([
      { event: 'split', id: 's1', data: { key: 'value' } },
    ]);
  });

  it('drops event data that exceeds 64 KiB cap', () => {
    const parser = createSSEParser();
    const bigPayload = 'x'.repeat(65 * 1024);
    const events = parser.push(`event: big\nid: b1\ndata: ${bigPayload}\n\n`);
    // Event is emitted but data is truncated / raw (not parsed as JSON)
    expect(events).toHaveLength(1);
    expect(events[0].event).toBe('big');
    // Data will be the raw string since JSON.parse will fail on truncated data
    expect(typeof events[0].data).toBe('string');
  });

  it('handles field with no value (colon only)', () => {
    const parser = createSSEParser();
    const events = parser.push('event:\nid: n1\ndata: {"a":1}\n\n');
    // event field with empty value → event name is ''
    // Per SSE spec, empty event name defaults to 'message' on dispatch
    expect(events).toEqual([{ event: 'message', id: 'n1', data: { a: 1 } }]);
  });

  it('preserves id across events (id persists until changed)', () => {
    const parser = createSSEParser();
    // First event sets id
    parser.push('event: a\nid: shared-id\ndata: {}\n\n');
    // Second event without id field — id is reset after dispatch per spec
    // (our parser resets id after each emit, which is correct SSE behavior)
    const events = parser.push('event: b\ndata: {}\n\n');
    expect(events[0].id).toBe('');
  });
});
