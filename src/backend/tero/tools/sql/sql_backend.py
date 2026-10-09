from abc import ABC, abstractmethod
from typing import Literal, Optional, get_args

from langchain_community.utilities import SQLDatabase

MaxStringLength = Literal[300, 1_000, 5_000, 10_000, 50_000, 100_000]
DEFAULT_MAX_STRING_LENGTH = get_args(MaxStringLength)[0]


class SqlBackend(ABC):
    @abstractmethod
    def list_databases(self) -> list[str]: ...

    @abstractmethod
    def list_schemas(self, database: str) -> list[str]: ...

    @abstractmethod
    def get_sql_database(
        self, name: str, schema: Optional[str] = None, max_string_length: Optional[int] = None
    ) -> SQLDatabase: ...
