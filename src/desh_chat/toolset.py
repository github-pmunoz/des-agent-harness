"""
The tools chat-des offers by default. Each is a plain function; Tool.define derives its schema
from the signature and Google-style docstring, so what the model is told and what the function
accepts cannot drift. A tool asks the operator before running unless it is registered with
confirm=False, which is a declaration that it is read-only.
"""
from datetime import datetime

from desh.tools import ToolRegistry


def current_time(timezone_offset_hours: int = 0) -> str:
    """Current date and time, as an ISO 8601 string.

    Args:
        timezone_offset_hours: Hours to add to the local clock (use 0 for local time).
    """
    from datetime import timedelta
    return (datetime.now().astimezone() + timedelta(hours=timezone_offset_hours)).isoformat(timespec="seconds")


def default_registry() -> ToolRegistry:
    return ToolRegistry().add(current_time, confirm=False)
