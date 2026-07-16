from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, field_validator
from sqlmodel import SQLModel, Field

from ..core.domain import CamelCaseModel


class ApiKey(SQLModel, table=True):
    __tablename__: Any = "api_key"
    id: int = Field(primary_key=True, default=None)
    user_id: int = Field(foreign_key="user.id", index=True)
    name: str = Field(max_length=100)
    key_id: str = Field(max_length=32, index=True, unique=True)
    hashed_secret: str = Field(max_length=256)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = Field(default=None)
    last_used_at: Optional[datetime] = Field(default=None)


class ApiKeyCreate(BaseModel):
    name: str
    expires_in_days: Optional[int] = None

    @field_validator("expires_in_days")
    @classmethod
    def _validate_expires_in_days(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("expires_in_days must be greater than 0")
        return value


class ApiKeyCreated(CamelCaseModel):
    id: int
    name: str
    api_key: str


class ApiKeyTokenRequest(BaseModel):
    api_key: str


class ApiKeyTokenResponse(CamelCaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
