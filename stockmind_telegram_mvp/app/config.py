import math
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _positive_int_setting(name: str, default: int) -> int:
    """Read a positive integer setting and fail at startup on invalid values."""
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer.")
    return value


def _positive_finite_float_setting(name: str, default: float) -> float:
    """Read a finite positive timeout and reject NaN, infinity, zero or negatives."""
    raw_value = os.getenv(name, str(default))
    try:
        value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{name} must be a positive finite number.") from exc
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{name} must be a positive finite number.")
    return value


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
MAX_CHART_IMAGE_BYTES = _positive_int_setting(
    "MAX_CHART_IMAGE_BYTES", 5 * 1024 * 1024
)
MAX_CONCURRENT_CHART_ANALYSES = int(os.getenv("MAX_CONCURRENT_CHART_ANALYSES", "2"))
if not 1 <= MAX_CONCURRENT_CHART_ANALYSES <= 10:
    raise RuntimeError("MAX_CONCURRENT_CHART_ANALYSES must be between 1 and 10.")
DAILY_QUERY_LIMIT = int(os.getenv("DAILY_QUERY_LIMIT", "10"))
BETA_USER_LIMIT = int(os.getenv("BETA_USER_LIMIT", "10"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
MAX_HISTORY_BYTES = _positive_int_setting("MAX_HISTORY_BYTES", 24 * 1024)
RESEARCH_TURN_TIMEOUT_SECONDS = _positive_finite_float_setting(
    "RESEARCH_TURN_TIMEOUT_SECONDS", 120.0
)
USER_LOCK_WAIT_TIMEOUT_SECONDS = _positive_finite_float_setting(
    "USER_LOCK_WAIT_TIMEOUT_SECONDS", 30.0
)
CHART_CAPACITY_WAIT_TIMEOUT_SECONDS = _positive_finite_float_setting(
    "CHART_CAPACITY_WAIT_TIMEOUT_SECONDS", 30.0
)
_DEFAULT_PREFERENCES_DB = Path(__file__).resolve().parents[1] / "data" / "user_preferences.sqlite3"
USER_PREFERENCES_DB = os.getenv(
    "STOCKMIND_PREFERENCES_DB",
    str(_DEFAULT_PREFERENCES_DB),
).strip() or str(_DEFAULT_PREFERENCES_DB)
