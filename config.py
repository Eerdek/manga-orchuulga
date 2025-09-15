from dotenv import load_dotenv
import os
load_dotenv()

TORII_API_KEY = os.getenv("TORII_API_KEY", "").strip()
TORII_API_URL = os.getenv("TORII_API_URL", "").strip()
TORII_API_HOST = os.getenv("TORII_API_HOST", "").strip()
TORII_API_TYPE = os.getenv("TORII_API_TYPE", "rapidapi").strip().lower()  # rapidapi | bearer

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

SOURCE_LANG = os.getenv("SOURCE_LANG", "auto").strip()
TARGET_LANG = os.getenv("TARGET_LANG", "mn").strip()

DEFAULT_FONT_PATH = os.getenv("FONT_PATH", "fonts/NotoSans-Regular.ttf")
