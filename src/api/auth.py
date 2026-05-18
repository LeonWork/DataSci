"""
src/api/auth.py
---------------
Small local authentication helpers for the custom web app.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import hmac
import os
import re
import time
from pathlib import Path

import bcrypt
from dotenv import load_dotenv

from src.utils.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
USER_STORE_PATH = (
    Path("/tmp") / "churnguard_app_users.json"
    if os.getenv("VERCEL")
    else ROOT / "data" / "app_users.json"
)
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")
DEFAULT_AUTH_USERNAME = "admin"
DEFAULT_AUTH_PASSWORD_HASH = b"$2b$12$amCWoXmqjip9GRVhRnmNJ.DBvO1ayDKDMK7aOceeiXAXP4kWdmS4m"
SESSION_TTL_SECONDS = 60 * 60 * 12
VALID_ROLES = {"owner", "analyst", "viewer"}


@dataclass(frozen=True)
class AuthenticatedUser:
    username: str
    email: str
    company_id: str
    role: str


def normalize_username(username: str) -> str:
    return username.strip().lower()


def normalize_company_id(company_id: str | None) -> str:
    value = (company_id or auth_setting("CHURNGUARD_COMPANY_ID") or "default").strip().lower()
    normalized = re.sub(r"[^a-z0-9_-]+", "-", value).strip("-")
    return normalized or "default"


def company_name() -> str:
    return (auth_setting("CHURNGUARD_COMPANY_NAME") or "ChurnGuard Pilot").strip()


def normalize_role(role: str | None, default: str = "analyst") -> str:
    value = (role or default).strip().lower()
    return value if value in VALID_ROLES else default


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


def truthy_setting(name: str, default: str = "") -> bool:
    return auth_setting(name, default).lower() in {"1", "true", "yes", "on"}


def signup_requires_invite() -> bool:
    return bool(auth_setting("CHURNGUARD_SIGNUP_CODE"))


def signup_enabled() -> bool:
    return truthy_setting("CHURNGUARD_ENABLE_SIGNUP") or signup_requires_invite()


def _session_secret() -> bytes:
    secret = (
        auth_setting("CHURNGUARD_SESSION_SECRET")
        or auth_setting("CHURNGUARD_PASSWORD_HASH")
        or auth_setting("CHURNGUARD_PASSWORD")
        or DEFAULT_AUTH_PASSWORD_HASH.decode()
    )
    return secret.encode()


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_session_token(username: str) -> str:
    user = {
        "username": username,
        "email": "",
        "company_id": normalize_company_id(None),
        "role": "owner",
    }
    return create_session_token_for_user(user)


def create_session_token_for_user(user: dict) -> str:
    payload = {
        "sub": normalize_username(str(user.get("username", ""))),
        "email": str(user.get("email", "")),
        "company_id": normalize_company_id(str(user.get("company_id", ""))),
        "role": normalize_role(str(user.get("role", "")), default="viewer"),
        "exp": int(time.time()) + SESSION_TTL_SECONDS,
    }
    payload_part = _b64encode(json.dumps(payload, separators=(",", ":")).encode())
    signature = hmac.new(_session_secret(), payload_part.encode(), "sha256").digest()
    return f"{payload_part}.{_b64encode(signature)}"


def verify_session_token(token: str) -> AuthenticatedUser | None:
    try:
        payload_part, signature_part = token.split(".", 1)
        expected = hmac.new(_session_secret(), payload_part.encode(), "sha256").digest()
        supplied = _b64decode(signature_part)
        if not hmac.compare_digest(expected, supplied):
            return None
        payload = json.loads(_b64decode(payload_part))
    except Exception:
        return None

    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    username = payload.get("sub")
    if not isinstance(username, str) or not USERNAME_PATTERN.match(username):
        return None
    return AuthenticatedUser(
        username=normalize_username(username),
        email=str(payload.get("email", "")),
        company_id=normalize_company_id(str(payload.get("company_id", ""))),
        role=normalize_role(str(payload.get("role", "")), default="viewer"),
    )


def create_user(
    username: str,
    email: str,
    password: str,
    confirm_password: str,
    invite_code: str | None = None,
) -> tuple[bool, str, dict | None]:
    expected_invite = auth_setting("CHURNGUARD_SIGNUP_CODE")
    if not signup_enabled():
        return False, "Account creation is disabled for this workspace.", None
    if expected_invite and not hmac.compare_digest((invite_code or "").strip(), expected_invite):
        return False, "Enter the workspace invite code.", None

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
        "company_id": normalize_company_id(None),
        "role": normalize_role(auth_setting("CHURNGUARD_SIGNUP_ROLE"), default="analyst"),
        "password_hash": hash_password(password),
    }
    users[username_key] = user
    save_users(users)
    return True, "Account created.", user


def authenticate_user(username: str, password: str) -> dict | None:
    username_key = normalize_username(username)
    user = load_users().get(username_key)
    if user and verify_password(password, user["password_hash"]):
        user.setdefault("company_id", normalize_company_id(None))
        user.setdefault("role", "analyst")
        return user

    default_admin_enabled = truthy_setting("CHURNGUARD_ENABLE_DEFAULT_ADMIN")
    expected_username = normalize_username(auth_setting("CHURNGUARD_USERNAME", DEFAULT_AUTH_USERNAME))
    password_hash = auth_setting("CHURNGUARD_PASSWORD_HASH")
    plain_password = auth_setting("CHURNGUARD_PASSWORD")

    if not (password_hash or plain_password or default_admin_enabled):
        return None

    username_ok = hmac.compare_digest(username_key, expected_username)
    if password_hash:
        password_ok = verify_password(password, password_hash)
    elif plain_password:
        password_ok = hmac.compare_digest(password, plain_password)
    else:
        password_ok = bcrypt.checkpw(password.encode(), DEFAULT_AUTH_PASSWORD_HASH)

    if username_ok and password_ok:
        return {
            "username": username_key,
            "email": "",
            "company_id": normalize_company_id(None),
            "role": "owner",
        }

    return None
