from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel import Session

from .db import get_session
from .models import User
from .security import decode_access_token

bearer = HTTPBearer(auto_error=False)


def current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    session: Annotated[Session, Depends(get_session)],
) -> User:
    if not credentials:
        raise HTTPException(status_code=401, detail="需要登录")
    try:
        user_id: UUID = decode_access_token(credentials.credentials)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="登录已失效") from exc
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user


CurrentUser = Annotated[User, Depends(current_user)]
