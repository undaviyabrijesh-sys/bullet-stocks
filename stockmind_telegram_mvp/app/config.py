import os
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
