"""Unit tests for API authentication and security."""

from roadvision.api.auth import (
    UserLogin,
    UserRegister,
    authenticate_user,
    register_user,
)


def test_preseeded_judge_login():
    token, user = authenticate_user(UserLogin(email="judge@innodata.com", password="Innodata2026!"))
    assert token
    assert user.role == "judge"
    assert user.name == "Innodata Hackathon Judge"


def test_preseeded_lead_login():
    token, user = authenticate_user(UserLogin(email="lead@roadvision.ai", password="RoadVision2026!"))
    assert token
    assert user.role == "admin"


def test_user_registration_and_login():
    reg = UserRegister(
        name="Reviewer Test",
        email="new_reviewer@innodata.com",
        password="SecurePass2026!",
        organization="MIT AI Lab",
        role="researcher",
    )
    user = register_user(reg)
    assert user.email == "new_reviewer@innodata.com"

    token, u = authenticate_user(UserLogin(email="new_reviewer@innodata.com", password="SecurePass2026!"))
    assert token
    assert u.name == "Reviewer Test"
