import hashlib
import time

def generate_ticket_id(initial_text: str, session_ts: int | None = None) -> str:
    """
    Диалоговый ticket_id: hash( normalized_text + session_start_ts )
    """
    text_norm = (initial_text or "").strip().lower()
    ts = str(session_ts or int(time.time()))
    return hashlib.md5((text_norm + "|" + ts).encode("utf-8")).hexdigest()[:12]
