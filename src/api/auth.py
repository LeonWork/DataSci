"""
src/api/auth.py
---------------
Small local authentication helpers for the custom web app.
"""

from __future__ import annotations

import json
import hmac
import os
import re
from pathlib import Path

import bcrypt

from src.utils.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
USER_STORE_PATH = (
    Path("/tmp") / "churnguard_app_users.json"
    if os.getenv("VERCEL")
    else ROOT / "data" / "app_users.json"
)
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")
DEFAULT_AUTH_USERNAME = "admin"
DEFAULT_AUTH_PASSWORD_HASH = b"$2b$12$amCWoXmqjip9GRVhRnmNJ.DBvO1ayDKDMK7aOceeiXAXP4kWdmS4m"


def normalize_username(username: str) -> str:
    return username.strip().lower()


def load_users() -> dict:
    if not USER_STORE_PATH.exists():
        return {}

    try:
        return json.loads(USER_STORE_PATH.read_text())
    except json.JSONDecodeError:
        logger.error("User store is not valid JSON: %s", USER_STORE_PATH)
        return {}


def save_users(users: dict) -> None:
    USER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_STORE_PATH.write_text(json.dumps(users, indent=2, sort_keys=True))


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def auth_setting(name: str, default: str | None = None) -> str | None:
    return os.getenv(name) or default


def create_user(username: str, email: str, password: str, confirm_password: str) -> tuple[bool, str, dict | None]:
    username_key = normalize_username(username)
    email = email.strip().lower()

    if not USERNAME_PATTERN.match(username_key):
        return False, "Use 3-32 characters: letters, numbers, dots, dashes, or underscores.", None
    if "@" not in email or "." not in email.split("@")[-1]:
        return False, "Enter a valid email address.", None
    if len(password) < 8:
        return False, "Password must be at least 8 characters.", None
    if password != confirm_password:
        return False, "Passwords do not match.", None

    users = load_users()
    if username_key in users:
        return False, "That username is already taken.", None
    if any(user.get("email") == email for user in users.values()):
        return False, "An account with that email already exists.", None

    user = {
        "username": username_key,
        "email": email,
        "password_hash": hash_password(password),
    }
    users[username_key] = user
    save_users(users)
    return True, "Account created.", user


def authenticate_user(username: str, password: str) -> dict | None:
    username_key = normalize_username(username)
    user = load_users().get(username_key)
    if user and verify_password(password, user["password_hash"]):
        return user

    expected_username = normalize_username(auth_setting("CHURNGUARD_USERNAME", DEFAULT_AUTH_USERNAME))
    password_hash = auth_setting("CHURNGUARD_PASSWORD_HASH")
    plain_password = auth_setting("CHURNGUARD_PASSWORD")

    username_ok = hmac.compare_digest(username_key, expected_username)
    if password_hash:
        password_ok = verify_password(password, password_hash)
    elif plain_password:
        password_ok = hmac.compare_digest(password, plain_password)
    else:
        password_ok = bcrypt.checkpw(password.encode(), DEFAULT_AUTH_PASSWORD_HASH)

    if username_ok and password_ok:
        return {"username": username_key, "email": ""}

    return None
