import threading

import pypdfium2

from .core import BasePdfProcessor


# pypdfium2 is not thread-safe; concurrent PdfDocument use can crash or corrupt state.
# https://pypdfium2.readthedocs.io/en/v4/python_api.html#thread-incompatibility
_pdfium_lock = threading.Lock()


class PyPdfiumPdfProcessor(BasePdfProcessor):

    def __init__(self):
        super().__init__(0.0)

    def _extract_pages_content(self, pdf_chunk: bytes, page_offset: int) -> dict[int, str]:
        pages_content = {}
        with _pdfium_lock, pypdfium2.PdfDocument(pdf_chunk) as pdf:
            for relative_page_number, page in enumerate(pdf, start=1):
                actual_page_number = relative_page_number + page_offset - 1
                try:
                    textpage = page.get_textpage()
                    try:
                        content = textpage.get_text_bounded()
                        pages_content[actual_page_number] = content.replace("\r", "").strip()
                    finally:
                        textpage.close()
                finally:
                    page.close()
        return pages_content
