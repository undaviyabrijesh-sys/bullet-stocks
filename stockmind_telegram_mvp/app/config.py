import math
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value == "replace_me":
        raise RuntimeError(f"Missing configuration: {name}. Set it in .env")
    return value


TELEGRAM_BOT_TOKEN = required("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = required("OPENAI_API_KEY")
TAPETIDE_TOKEN = required("TAPETIDE_TOKEN")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
# Keep this vision-capable; the default model accepts image input.
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", OPENAI_MODEL)
MAX_CHART_IMAGE_BYTES = int(os.getenv("MAX_CHART_IMAGE_BYTES", str(5 * 1024 * 1024)))
DAILY_QUERY_LIMIT = int(os.getenv("DAILY_QUERY_LIMIT", "10"))
BETA_USER_LIMIT = int(os.getenv("BETA_USER_LIMIT", "10"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
MAX_HISTORY_BYTES = int(os.getenv("MAX_HISTORY_BYTES", str(24 * 1024)))
if MAX_HISTORY_BYTES <= 0:
    raise RuntimeError("MAX_HISTORY_BYTES must be a positive integer.")
RESEARCH_TURN_TIMEOUT_SECONDS = float(os.getenv("RESEARCH_TURN_TIMEOUT_SECONDS", "120"))
USER_LOCK_WAIT_TIMEOUT_SECONDS = float(os.getenv("USER_LOCK_WAIT_TIMEOUT_SECONDS", "30"))
if not math.isfinite(USER_LOCK_WAIT_TIMEOUT_SECONDS) or USER_LOCK_WAIT_TIMEOUT_SECONDS <= 0:
    raise RuntimeError("USER_LOCK_WAIT_TIMEOUT_SECONDS must be a positive finite number.")
_DEFAULT_PREFERENCES_DB = Path(__file__).resolve().parents[1] / "data" / "user_preferences.sqlite3"
USER_PREFERENCES_DB = os.getenv(
    "STOCKMIND_PREFERENCES_DB",
    str(_DEFAULT_PREFERENCES_DB),
).strip() or str(_DEFAULT_PREFERENCES_DB)
