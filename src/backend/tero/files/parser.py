import asyncio
import logging

from ..files.domain import File, FileProcessor
from .core import BaseFileProcessor, FileQuota
from .domain import File, FileProcessor
from .processors.docx import DocxFileProcessor
from .processors.plaintext import PlainTextFileProcessor
from .processors.spreadsheet import XlsxFileProcessor, XlsFileProcessor
from .processors.image import ImageFileProcessor
from .processors.pdf import build_basic_pdf_processor, build_enhanced_pdf_processor


logger = logging.getLogger(__name__)


class UnsupportedFileError(Exception):
    def __init__(self, file_name: str):
        super().__init__(f"Unsupported file type: {file_name}")


async def extract_file_text(file: File, file_quota: FileQuota) -> str:
    processor = _find_file_processor(file)
    return await asyncio.to_thread(processor.extract_text, file, file_quota)


def _find_file_processor(file: File) -> BaseFileProcessor:
    processors = [
        PlainTextFileProcessor(),
        DocxFileProcessor(),
        XlsxFileProcessor(),
        XlsFileProcessor(),
        ImageFileProcessor(),
        build_basic_pdf_processor() if file.file_processor == FileProcessor.BASIC else build_enhanced_pdf_processor()
    ]
    found = next((processor for processor in processors if processor.supports(file)), None)
    if found is None:
        raise UnsupportedFileError(file.name)
    return found
