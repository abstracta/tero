from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlmodel import Field

from ..core.domain import CamelCaseModel


class FileStatus(Enum):
    PENDING = "PENDING"
    PROCESSED = "PROCESSED"
    ERROR = "ERROR"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    RATE_LIMITED = "RATE_LIMITED"


class FileProcessor(Enum):
    BASIC = 'BASIC'
    ENHANCED = 'ENHANCED'


class FileUpdate(CamelCaseModel):
    name: Optional[str] = None
    content_type: Optional[str] = None
    content: Optional[bytes] = None
    status: Optional[FileStatus] = None
    user_id: Optional[int] = None


class File(CamelCaseModel, table=True):
    id: int = Field(primary_key=True, default=None)
    name: str = Field(max_length=200)
    content_type: str = Field(max_length=100)
    user_id: int = Field(foreign_key="user.id")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)
    content: bytes
    status: FileStatus = Field(default=FileStatus.PENDING, index=True)
    processed_content: Optional[str] = Field(default=None)
    file_processor: FileProcessor = Field(default=FileProcessor.BASIC)

    def clone(self, user_id: int) -> 'File':
        return File(
            name=self.name,
            content_type=self.content_type,
            user_id=user_id,
            content=self.content,
            status=self.status,
            processed_content=self.processed_content,
            file_processor=self.file_processor
        )
    
    def update_with(self, update: FileUpdate):
        update_dict = update.model_dump(exclude_none=True)
        self.sqlmodel_update(update_dict)
        self.timestamp = datetime.now(timezone.utc)


class FileMetadata(CamelCaseModel, table=False):
    id: int
    name: str 
    content_type: str 
    user_id: int
    timestamp: datetime 
    status: FileStatus 
    file_processor: FileProcessor

    @staticmethod
    def from_file(file: File) -> 'FileMetadata':
        return FileMetadata.model_validate(file)


class FileMetadataWithContent(FileMetadata, table=False):
    processed_content: Optional[str] = None

    @staticmethod
    def from_file(file: File) -> 'FileMetadataWithContent':
        return FileMetadataWithContent.model_validate(file)
