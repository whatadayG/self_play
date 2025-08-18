# Makes `dialop.utils` a proper subpackage.
# Provide minimal utilities needed across the codebase.

from typing import Any

# Lightweight tokenizer length utility compatible with original utils.py
try:
    import tiktoken as _tiktoken
    _enc = _tiktoken.get_encoding("cl100k_base")
    def num_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    # Fallback if tiktoken unavailable; approximate by whitespace split
    def num_tokens(text: str) -> int:  # type: ignore[no-redef]
        return len(text.split())

# Minimal implementations to avoid circular import on dialop.utils
from collections import defaultdict
import time

SEP = "=" * 20
SEP1 = "-" * 20

def count_words(action_log: Any) -> int:
    count = 0
    try:
        for msg in action_log:
            if msg.get("type") == "message":
                count += len(str(msg.get("message", {}).get("data", "")).split())
    except Exception:
        pass
    return count


def retry(max_attempts: int = 3, delay: float = 60.0, allowed_exceptions: Any = None):
    if allowed_exceptions is None:
        allowed_exceptions = []
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if type(e) in allowed_exceptions:
                        print(f"Attempt {attempts+1} failed: {type(e)} {e}")
                        attempts += 1
                        time.sleep(delay)
                    else:
                        raise e
            print("Max attempts reached. Giving up.")
        return wrapper
    return decorator


class Logger:
    def __init__(self, logfile: str):
        self.logfile = logfile
        self.buffer = ""
        self.logs = defaultdict(list)
    def write(self, title: str = "", value: str = ""):
        self.buffer += f"\n\n{SEP} {title} {SEP}\n\n>{value}<"
    def log(self, key: str, value: str, title: str = ""):
        self.logs[key].append(f"\n\n{SEP1} {title} {SEP1}\n>{value}")
    def flush(self):
        with open(self.logfile, "a") as f:
            f.write(self.buffer)
        self.buffer = ""
    def flush_key(self, key: str, title: str = "", joiner: str = ""):
        with open(self.logfile, "a") as f:
            f.write(f"\n\n{SEP} {title} {SEP}\n{joiner.join(self.logs[key])}")
        self.logs[key] = []
    def close(self):
        self.flush()

# Re-export reconstruct_game_state when available
try:
    from .reconstruct_opt_game import reconstruct_game_state  # noqa: F401
except Exception:
    pass 