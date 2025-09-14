from datetime import datetime

def get_current_timestamp() -> str:
    """Return current timestamp in ISO format."""
    return datetime.utcnow().isoformat()

def parse_timestamp(ts: str) -> datetime:
    """Convert ISO string to datetime object."""
    return datetime.fromisoformat(ts)