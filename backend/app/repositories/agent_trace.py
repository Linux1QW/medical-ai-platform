"""Repository for agent trace event persistence and querying."""

from datetime import datetime
from typing import Optional

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_trace_event import AgentTraceEvent


class AgentTraceRepository:
    """Handles persistence and querying of agent trace events."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def append_event(
        self,
        trace_id: str,
        session_id: str,
        event_type: str,
        status: str,
        agent_name: Optional[str] = None,
        node_name: Optional[str] = None,
        input_payload: Optional[dict] = None,
        output_payload: Optional[dict] = None,
        input_hmac: Optional[str] = None,
        output_hmac: Optional[str] = None,
        error_code: Optional[str] = None,
        metadata_json: Optional[dict] = None,
    ) -> AgentTraceEvent:
        """Append a trace event with auto-allocated sequence number."""
        # Get next sequence for this trace
        seq_stmt = (
            select(func.coalesce(func.max(AgentTraceEvent.sequence), 0))
            .where(AgentTraceEvent.trace_id == trace_id)
        )
        result = await self.db.execute(seq_stmt)
        max_seq: int = result.scalar() or 0
        next_seq = max_seq + 1

        # Compute char counts
        input_char_count = None
        output_char_count = None
        if input_payload is not None:
            input_char_count = len(str(input_payload))
        if output_payload is not None:
            output_char_count = len(str(output_payload))

        event = AgentTraceEvent(
            trace_id=trace_id,
            sequence=next_seq,
            session_id=session_id,
            event_type=event_type,
            agent_name=agent_name,
            node_name=node_name,
            status=status,
            input_payload=input_payload,
            output_payload=output_payload,
            input_hmac=input_hmac,
            output_hmac=output_hmac,
            input_char_count=input_char_count,
            output_char_count=output_char_count,
            error_code=error_code,
            metadata_json=metadata_json,
        )
        self.db.add(event)
        await self.db.flush()
        return event

    async def list_events_by_trace(
        self,
        trace_id: str,
        limit: int = 500,
    ) -> list[AgentTraceEvent]:
        """List all events for a given trace, ordered by sequence."""
        stmt = (
            select(AgentTraceEvent)
            .where(AgentTraceEvent.trace_id == trace_id)
            .order_by(AgentTraceEvent.sequence.asc())
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def list_events_by_session(
        self,
        session_id: str,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 500,
    ) -> list[AgentTraceEvent]:
        """Query trace events by session, optionally filtered by type and time."""
        conditions = [AgentTraceEvent.session_id == session_id]
        if event_type is not None:
            conditions.append(AgentTraceEvent.event_type == event_type)
        if since is not None:
            conditions.append(AgentTraceEvent.created_at >= since)

        stmt = (
            select(AgentTraceEvent)
            .where(and_(*conditions))
            .order_by(AgentTraceEvent.created_at.asc())
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def count_events_by_trace(self, trace_id: str) -> int:
        """Count total events for a trace."""
        stmt = (
            select(func.count(AgentTraceEvent.id))
            .where(AgentTraceEvent.trace_id == trace_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar() or 0
