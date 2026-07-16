from ...core import BaseFileProcessor
from .pypdfium import PyPdfiumPdfProcessor
from .azure_document_intelligence import AzureDocumentIntelligencePdfProcessor
from .amazon_textract import AmazonTextractPdfProcessor


def build_basic_pdf_processor() -> BaseFileProcessor:
    return PyPdfiumPdfProcessor()


def is_enhanced_pdf_processor_available() -> bool:
    return AzureDocumentIntelligencePdfProcessor.is_configured() or AmazonTextractPdfProcessor.is_configured()


def build_enhanced_pdf_processor() -> BaseFileProcessor:
    if AzureDocumentIntelligencePdfProcessor.is_configured():
        return AzureDocumentIntelligencePdfProcessor()
    elif AmazonTextractPdfProcessor.is_configured():
        return AmazonTextractPdfProcessor()
    else:
        raise RuntimeError("No enhanced PDF processor available")
