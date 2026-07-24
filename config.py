import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
