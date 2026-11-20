"""Authentication, authorization, and security layer for RoadVision AI.

Provides JWT issuance, bcrypt password hashing, and role-based access control.
Includes instant demo profiles for hackathon judges and researchers.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field

# Secret key for JWT signing (for demonstration/hackathon environment)
SECRET_KEY = "roadvision-ai-hackathon-2026-secret-key-super-secure"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

security = HTTPBearer(auto_error=False)


class UserRegister(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=6)
    organization: str = Field(default="Innodata Hackathon")
    role: str = Field(default="evaluator")


class UserLogin(BaseModel):
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict[str, Any]


@dataclass
class UserRecord:
    name: str
    email: str
    password_hash: bytes
    role: str
    organization: str
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


def _hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=10))


def _verify_password(password: str, hashed: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed)
    except Exception:
        return False


# In-memory secure user store pre-seeded with demo and judge credentials
_USERS_DB: dict[str, UserRecord] = {
    "judge@innodata.com": UserRecord(
        name="Innodata Hackathon Judge",
        email="judge@innodata.com",
        password_hash=_hash_password("Innodata2026!"),
        role="judge",
        organization="Innodata Review Committee",
    ),
    "lead@roadvision.ai": UserRecord(
        name="Lead AI Engineer",
        email="lead@roadvision.ai",
        password_hash=_hash_password("RoadVision2026!"),
        role="admin",
        organization="RoadVision AI Core Team",
    ),
    "demo@roadvision.ai": UserRecord(
        name="Guest Evaluator",
        email="demo@roadvision.ai",
        password_hash=_hash_password("Demo2026!"),
        role="evaluator",
        organization="Independent AI Auditor",
    ),
}


def create_access_token(data: dict[str, Any], expires_delta: datetime.timedelta | None = None) -> str:
    to_encode = data.copy()
    now = datetime.datetime.now(datetime.timezone.utc)
    expire = now + (expires_delta or datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": now})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def register_user(reg: UserRegister) -> UserRecord:
    email_clean = reg.email.strip().lower()
    if email_clean in _USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )
    user = UserRecord(
        name=reg.name.strip(),
        email=email_clean,
        password_hash=_hash_password(reg.password),
        role=reg.role,
        organization=reg.organization.strip(),
    )
    _USERS_DB[email_clean] = user
    return user


def authenticate_user(cred: UserLogin) -> tuple[str, UserRecord]:
    email_clean = cred.email.strip().lower()
    user = _USERS_DB.get(email_clean)
    if not user or not _verify_password(cred.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({
        "sub": user.email,
        "name": user.name,
        "role": user.role,
        "org": user.organization,
    })
    return token, user


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> UserRecord | None:
    if not credentials:
        return None
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email or email not in _USERS_DB:
            return None
        return _USERS_DB[email]
    except jwt.PyJWTError:
        return None


def require_user(user: UserRecord | None = Depends(get_current_user)) -> UserRecord:
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to perform this action.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
