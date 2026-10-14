from io import BytesIO

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

from ..core import BaseFileProcessor, FileQuota
from ..domain import File


class DocxFileProcessor(BaseFileProcessor):

    def supports(self, file: File) -> bool:
        return file.name.lower().endswith('.docx')

    def extract_text(self, file: File, file_quota: FileQuota) -> str:
        document = docx.Document(BytesIO(file.content))
        parts = []
        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                if block.text.strip():
                    parts.append(block.text)
            elif isinstance(block, Table):
                table_text = self._format_table(block)
                if table_text:
                    parts.append(table_text)
        return "\n\n".join(parts)

    def _format_table(self, table: Table) -> str:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(" | ".join(cells))
        return "\n".join(rows) if rows else ""
