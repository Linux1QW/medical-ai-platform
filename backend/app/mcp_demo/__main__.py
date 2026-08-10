"""Entry point for MCP demo server.

Usage:
    python -m app.mcp_demo

Starts a JSON-RPC 2.0 stdio server exposing read-only medical tools.
Does NOT bind any network port or connect to production databases.
"""
from app.mcp_demo.server import run_stdio_server

if __name__ == "__main__":
    run_stdio_server()
