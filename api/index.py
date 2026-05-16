"""
Vercel entrypoint.

The actual application lives in src.api.main so local development and tests keep
using the same FastAPI app object.
"""

from src.api.main import app
