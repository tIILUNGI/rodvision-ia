"""Authentication routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from ..auth import (
    TokenResponse,
    UserLogin,
    UserRecord,
    UserRegister,
    authenticate_user,
    get_current_user,
    register_user,
    require_user,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
def route_register(req: UserRegister):
    user = register_user(req)
    token, _ = authenticate_user(UserLogin(email=req.email, password=req.password))
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "name": user.name,
            "email": user.email,
            "role": user.role,
            "organization": user.organization,
        },
    }


@router.post("/login", response_model=TokenResponse)
def route_login(cred: UserLogin):
    token, user = authenticate_user(cred)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "name": user.name,
            "email": user.email,
            "role": user.role,
            "organization": user.organization,
        },
    }


@router.get("/me")
def route_me(user: UserRecord = Depends(require_user)):
    return {
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "organization": user.organization,
        "created_at": user.created_at,
    }
