import base64
import hashlib
import os
from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt
from argon2 import PasswordHasher
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .config import get_settings

password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return password_hasher.verify(password_hash, password)
    except Exception:
        return False


def create_access_token(user_id: UUID) -> str:
    settings = get_settings()
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(UTC) + timedelta(days=30),
        "iat": datetime.now(UTC),
    }
    return jwt.encode(payload, settings.secret_key, algorithm="HS256")


def decode_access_token(token: str) -> UUID:
    payload = jwt.decode(token, get_settings().secret_key, algorithms=["HS256"])
    return UUID(payload["sub"])


def _aes_key() -> bytes:
    return hashlib.sha256(get_settings().encryption_key.encode("utf-8")).digest()


def encrypt_secret(value: str) -> str:
    nonce = os.urandom(12)
    cipher = AESGCM(_aes_key()).encrypt(nonce, value.encode("utf-8"), None)
    return base64.urlsafe_b64encode(nonce + cipher).decode("ascii")


def decrypt_secret(value: str) -> str:
    raw = base64.urlsafe_b64decode(value.encode("ascii"))
    return AESGCM(_aes_key()).decrypt(raw[:12], raw[12:], None).decode("utf-8")
