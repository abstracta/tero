from abc import ABC, abstractmethod
from typing import Optional

from langchain_community.utilities import SQLDatabase


class SqlBackend(ABC):
    @abstractmethod
    def list_databases(self) -> list[str]: ...

    @abstractmethod
    def list_schemas(self, database: str) -> list[str]: ...

    @abstractmethod
    def get_sql_database(self, name: str, schema: Optional[str] = None) -> SQLDatabase: ...
