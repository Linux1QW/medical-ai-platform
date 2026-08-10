/**
 * Split-chunk SSE parser.
 *
 * Owns event / data / id state across network chunks.
 * Supports CRLF and LF line endings, multi-line `data:` fields,
 * and caps a single event at 64 KiB to prevent memory blow-up.
 */

export interface SSEEvent {
  event: string;
  id: string;
  data: unknown; // parsed JSON when possible, raw string otherwise
}

const MAX_EVENT_BYTES = 64 * 1024; // 64 KiB

export function createSSEParser(): { push(chunk: string): SSEEvent[]; flush(): SSEEvent[] } {
  let buffer = '';
  let currentEvent = '';
  let currentData: string[] = [];
  let currentId = '';
  let byteCount = 0;

  /** Try to emit an event when a blank line is encountered. */
  function tryEmit(): SSEEvent | null {
    if (!currentEvent && currentData.length === 0) return null;

    const event = currentEvent || 'message';
    const rawData = currentData.join('\n');

    let parsed: unknown = rawData;
    try {
      parsed = JSON.parse(rawData);
    } catch {
      // keep as raw string
    }

    const emitted: SSEEvent = {
      event,
      id: currentId,
      data: parsed,
    };

    // Reset per-event state
    currentEvent = '';
    currentData = [];
    currentId = '';
    byteCount = 0;

    return emitted;
  }

  /**
   * Process a single logical line (after CRLF → LF normalisation).
   * Returns an emitted event if the line was a terminating blank line.
   */
  function processLine(line: string): SSEEvent | null {
    if (line === '') {
      // Blank line → dispatch event
      return tryEmit();
    }

    // Guard: cap event size
    if (byteCount >= MAX_EVENT_BYTES) {
      // Discard further data for this oversized event
      return null;
    }
    byteCount += line.length;

    if (line.startsWith(':')) {
      // Comment line — ignore
      return null;
    }

    let field: string;
    let value: string;
    const colonIdx = line.indexOf(':');
    if (colonIdx === -1) {
      field = line;
      value = '';
    } else {
      field = line.slice(0, colonIdx);
      // Strip a single leading space from value per SSE spec
      value = line.slice(colonIdx + 1).replace(/^ /, '');
    }

    switch (field) {
      case 'event':
        currentEvent = value;
        break;
      case 'data':
        currentData.push(value);
        break;
      case 'id':
        currentId = value;
        break;
      case 'retry':
        // Not used by this parser, but acknowledged
        break;
      default:
        // Unknown field — ignore per spec
        break;
    }

    return null;
  }

  return {
    push(chunk: string): SSEEvent[] {
      // Normalise CRLF → LF, then append to buffer
      buffer += chunk.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

      const lines = buffer.split('\n');
      // The last element is the (possibly incomplete) trailing line — keep in buffer
      buffer = lines.pop() ?? '';

      const events: SSEEvent[] = [];
      for (const line of lines) {
        const emitted = processLine(line);
        if (emitted) events.push(emitted);
      }
      return events;
    },

    flush(): SSEEvent[] {
      // Process any remaining content in the buffer
      if (buffer === '' && currentData.length === 0 && !currentEvent) return [];

      const events: SSEEvent[] = [];
      if (buffer !== '') {
        const emitted = processLine(buffer);
        if (emitted) events.push(emitted);
        buffer = '';
      }
      // If there is accumulated event/data state without a trailing blank line,
      // emit it as a best-effort final event.
      if (currentEvent || currentData.length > 0) {
        const emitted = tryEmit();
        if (emitted) events.push(emitted);
      }
      return events;
    },
  };
}
