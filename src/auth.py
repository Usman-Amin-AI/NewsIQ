from typing import Dict

from src.config import AppConfig


def authenticate_user(config: AppConfig, username: str, password: str) -> bool:
    if not username or not password:
        return False

    for valid_username, valid_password in config.auth_user_pairs():
        if username.strip() == valid_username and password == valid_password:
            return True

    return False


def build_user_session(config: AppConfig, username: str) -> Dict[str, str]:
    normalized_username = username.strip()
    user_id = normalized_username.lower().replace(" ", "_")
    return {
        "username": normalized_username,
        "user_id": user_id,
        "is_admin": normalized_username in config.admin_user_list(),
    }
