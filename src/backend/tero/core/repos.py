from typing import Any, AsyncGenerator, Optional, cast

from cryptography.fernet import Fernet
from sqlalchemy import Dialect
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import QueryableAttribute
from sqlmodel import Column, Field, String, TypeDecorator
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel.sql.expression import SelectOfScalar

from .env import env


engine = create_async_engine(env.db_url)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session


def plain_postgresql_url() -> str:
    return engine.url.set(drivername="postgresql").render_as_string(hide_password=False)


# this method allows to easily cast a query to SelectOfScalar to avoid type errors when using sqlmodel exec method
# for example when using delete statements
# this is related to https://github.com/fastapi/sqlmodel/issues/909
def scalar(val: Any) -> SelectOfScalar:
    return cast(SelectOfScalar, val)


# this method allows to easily cast a value to QueryableAttribute to avoid type errors when using sqlmodel selectinload method
def attr(val: Any) -> QueryableAttribute:
    return val


class _EncryptedStrTypeDecorator(TypeDecorator):
    impl = String
    cache_ok = True

    def _get_cipher(self) -> Fernet:
        return Fernet(env.secret_encryption_key.get_secret_value().encode())

    def process_bind_param(self, value: Optional[str], dialect: Dialect) -> Optional[str]:
        return self._get_cipher().encrypt(value.encode()).decode() if value is not None else None

    def process_result_value(self, value: Optional[str], dialect: Dialect) -> Optional[str]:
        return self._get_cipher().decrypt(value.encode()).decode() if value is not None else None


def EncryptedField(nullable: bool = False, default: Optional[str] = None, **kwargs) -> Field: # type: ignore[reportUnknownReturnType]
    return Field(sa_column=Column(_EncryptedStrTypeDecorator, nullable=nullable, default=default), **kwargs)
