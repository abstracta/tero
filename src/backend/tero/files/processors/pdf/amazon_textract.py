from collections import defaultdict
from dataclasses import dataclass
import logging
import time
from typing import cast, Optional
import uuid

from tabulate import tabulate

from ....core.env import env
from ....ai_models.aws_provider import build_aws_service_client
from .core import BasePdfProcessor


logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_block(cls, block: dict) -> Optional["BoundingBox"]:
        bbox = block.get("Geometry", {}).get("BoundingBox", {})
        if not bbox:
            return None
        return cls(
            x=bbox.get("Left", 0.0),
            y=bbox.get("Top", 0.0),
            width=bbox.get("Width", 0.0),
            height=bbox.get("Height", 0.0)
        )

    def contains(self, other: "BoundingBox") -> bool:
        return (
            other.x >= self.x
            and other.x + other.width <= self.x + self.width
            and other.y >= self.y
            and other.y + other.height <= self.y + self.height
        )

    def contains_point(self, x: float, y: float) -> bool:
        return (self.x <= x <= self.x + self.width) and (self.y <= y <= self.y + self.height)


@dataclass
class BoundedElement:
    content: str
    y: float
    height: float
    bbox: Optional[BoundingBox] = None


@dataclass
class BoundedParagraph(BoundedElement):

    @classmethod
    def from_line(cls, line_block: dict) -> Optional["BoundedParagraph"]:
        content = line_block.get("Text", "").strip()
        if not content:
            return None
        bbox = BoundingBox.from_block(line_block)
        if bbox:
            return cls(content=content, y=bbox.y, height=bbox.height, bbox=bbox)
        return cls(content=content, y=0.0, height=0.0, bbox=None)


@dataclass
class BoundedTable(BoundedElement):

    @classmethod
    def from_table(
        cls,
        table_block: dict,
        block_map: dict,
        page_blocks: Optional[list[dict]] = None
    ) -> Optional["BoundedTable"]:
        bbox = BoundingBox.from_block(table_block)
        if not bbox:
            return None
        grid = cls._create_grid_from_table(table_block, block_map, page_blocks)
        content = cls._format_grid_as_markdown(grid)
        if not content.strip():
            return None
        return cls(content=content, y=bbox.y, height=bbox.height, bbox=bbox)

    @classmethod
    def _create_grid_from_table(
        cls,
        table_block: dict,
        block_map: dict,
        page_blocks: Optional[list[dict]] = None
    ) -> list[list[str]]:
        cells = cls._get_table_cells(table_block, block_map)
        if not cells:
            return []

        max_row = max((cell.get("RowIndex", 1) for cell in cells), default=1)
        max_col = max((cell.get("ColumnIndex", 1) for cell in cells), default=1)
        grid = [["" for _ in range(max_col)] for _ in range(max_row)]

        for cell in cells:
            row = cell.get("RowIndex", 1) - 1
            col = cell.get("ColumnIndex", 1) - 1
            grid[row][col] = cls._normalize_cell_text(cls._extract_cell_text(cell, block_map, page_blocks))

        return grid

    @classmethod
    def _get_table_cells(cls, table_block: dict, block_map: dict) -> list[dict]:
        cell_ids = cls._get_relationship_ids(table_block, "CHILD")
        cells = [block_map.get(cell_id) for cell_id in cell_ids]
        return [cell for cell in cells if cell and cell.get("BlockType") == "CELL"]

    @staticmethod
    def _get_relationship_ids(block: dict, relation_type: str) -> list[str]:
        ids: list[str] = []
        for relation in block.get("Relationships", []):
            if relation.get("Type") == relation_type:
                ids.extend(relation.get("Ids", []))
        return ids

    @classmethod
    def _extract_cell_text(cls, cell_block: dict, block_map: dict, page_blocks: Optional[list[dict]]) -> str:
        ret = cls._extract_cell_text_from_children(cell_block, block_map)
        return ret if ret else cls._extract_cell_text_from_bounding_box(cell_block, page_blocks)

    @classmethod
    def _extract_cell_text_from_children(cls, cell_block: dict, block_map: dict) -> str:
        ret = []
        for child_id in cls._get_relationship_ids(cell_block, "CHILD"):
            child_block = block_map.get(child_id)
            if not child_block:
                continue
            if child_block.get("BlockType") == "WORD":
                ret.append(child_block.get("Text", ""))
            elif child_block.get("BlockType") == "SELECTION_ELEMENT":
                status = child_block.get("SelectionStatus")
                if status == "SELECTED":
                    ret.append("X")
        return " ".join(part for part in ret if part).strip()

    @classmethod
    def _extract_cell_text_from_bounding_box(cls, cell_block: dict, page_blocks: Optional[list[dict]]) -> str:
        cell_bbox = BoundingBox.from_block(cell_block)
        if not page_blocks or not cell_bbox:
            return ""
        ret = []
        for block in page_blocks:
            block_type = block.get("BlockType")
            if block_type == "WORD" or block_type == "SELECTION_ELEMENT":
                word_bbox = BoundingBox.from_block(block)
                if not word_bbox:
                    continue
                center_x = word_bbox.x + word_bbox.width / 2
                center_y = word_bbox.y + word_bbox.height / 2
                if cell_bbox.contains_point(center_x, center_y):
                    if block_type == "WORD":
                        ret.append((word_bbox.y, word_bbox.x, block.get("Text", "")))
                    if block.get("SelectionStatus") == "SELECTED":
                        ret.append((word_bbox.y, word_bbox.x, "X"))
        ret.sort(key=lambda item: (item[0], item[1]))
        return " ".join(text for _, __, text in ret if text).strip()

    @staticmethod
    def _normalize_cell_text(text: str) -> str:
        return text.replace("\n", "<br/>").strip()

    @staticmethod
    def _format_grid_as_markdown(grid: list[list[str]]) -> str:
        if not grid:
            return ""
        header, *data = grid
        table = tabulate(data, headers=header, tablefmt="pipe")
        return f"\n{table}\n"


# We use async API because it supports multi-page PDFs
class AmazonTextractPdfProcessor(BasePdfProcessor):

    @staticmethod
    def is_configured() -> bool:
        return bool(env.aws_s3_bucket)

    def __init__(self):
        super().__init__(cast(float, env.aws_textract_cost_per_1k_pages_usd))
        self._textract_client = build_aws_service_client("textract")
        self._s3_client = build_aws_service_client("s3")
        self._s3_bucket = cast(str, env.aws_s3_bucket)


    def _extract_pages_content(self, pdf_chunk: bytes, page_offset: int) -> dict[int, str]:
        s3_key = f"textract/{uuid.uuid4()}.pdf"
        try:
            self._s3_client.put_object(
                Bucket=self._s3_bucket,
                Key=s3_key,
                Body=pdf_chunk,
                ContentType="application/pdf")

            start_response = self._textract_client.start_document_analysis(
                DocumentLocation={
                    "S3Object": {
                        "Bucket": self._s3_bucket,
                        "Name": s3_key
                    }
                },
                FeatureTypes=["TABLES"]
            )
            job_id = start_response["JobId"]
            pages_content = self._poll_and_extract_results(job_id, page_offset)
            return pages_content
        finally:
            try:
                self._s3_client.delete_object(Bucket=self._s3_bucket, Key=s3_key)
            except Exception:
                logger.warning(f"Textract: Failed to cleanup S3 object {s3_key}", exc_info=True)

    def _poll_and_extract_results(self, job_id: str, page_offset: int) -> dict[int, str]:
        pages_content: dict[int, str] = {}
        max_wait_seconds = 600
        deadline = time.time() + max_wait_seconds
        while time.time() < deadline:
            analysis_response = self._textract_client.get_document_analysis(JobId=job_id)
            status = analysis_response["JobStatus"]
            if status == "SUCCEEDED":
                pages_content.update(self._extract_text_from_response(analysis_response, page_offset))

                # Handle pagination
                next_token = analysis_response.get("NextToken")
                while next_token:
                    paginated_response = self._textract_client.get_document_analysis(JobId=job_id, NextToken=next_token)
                    pages_content.update(self._extract_text_from_response(paginated_response, page_offset))
                    next_token = paginated_response.get("NextToken")

                return pages_content

            elif status == "FAILED":
                error_message = analysis_response.get("StatusMessage", "Unknown error")
                raise RuntimeError(f"Textract job {job_id} failed: {error_message}")

            elif status in ["IN_PROGRESS", "PARTIAL_SUCCESS"]:
                time.sleep(3)
            else:
                raise RuntimeError(f"Textract job {job_id} returned unexpected status: {status}")
        raise TimeoutError(f"Textract job {job_id} exceeded max wait time of {max_wait_seconds} seconds")

    def _extract_text_from_response(self, response: dict, page_offset: int) -> dict[int, str]:
        blocks = response.get("Blocks", [])
        block_map = {block.get("Id"): block for block in blocks if block.get("Id")}
        ret: dict[int, str] = {}
        for page_num, page_blocks in self._group_blocks_by_page(blocks, page_offset).items():
            elements = self._create_page_elements(page_blocks, block_map)
            ret[page_num] = self._combine_elements_content(elements)
        return ret

    def _group_blocks_by_page(self, blocks: list[dict], page_offset: int) -> dict[int, list[dict]]:
        ret: dict[int, list[dict]] = defaultdict(list)
        for block in blocks:
            ret[block.get("Page", 1) + page_offset - 1].append(block)
        return dict(ret)

    def _create_page_elements(self, page_blocks: list[dict], block_map: dict) -> list[BoundedElement]:
        paragraph_elements = self._create_page_elements_by_type(
            "LINE",
            BoundedParagraph.from_line,
            page_blocks
        )
        table_elements = self._create_page_elements_by_type(
            "TABLE",
            BoundedTable.from_table,
            page_blocks,
            block_map,
            page_blocks
        )
        ret: list[BoundedElement] = []
        for paragraph in paragraph_elements:
            if paragraph.bbox and not any(table.bbox and table.bbox.contains(paragraph.bbox) for table in table_elements):
                ret.append(paragraph)
            elif not paragraph.bbox:
                ret.append(paragraph)
        ret.extend(table_elements)
        return ret

    def _create_page_elements_by_type(self, element_type: str, factory, page_blocks: list[dict], *factory_args) -> list[BoundedElement]:
        ret: list[BoundedElement] = []
        for element in self._get_page_elements(page_blocks, element_type):
            created = factory(element, *factory_args)
            if created:
                ret.append(cast(BoundedElement, created))
        return ret

    def _get_page_elements(self, page_blocks: list[dict], element_type: str) -> list[dict]:
        return [block for block in page_blocks if block.get("BlockType") == element_type]


    def _combine_elements_content(self, elements: list[BoundedElement]) -> str:
        elements.sort(key=lambda x: x.y)
        return "\n".join(element.content for element in elements)
