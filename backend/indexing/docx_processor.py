"""Word(.docx) 结构化提取：按 Heading 1/2 章节拆成多「页」供三层分块复用。"""
from __future__ import annotations

import re
from typing import Any

from docx import Document as DocxDocument
from langchain_core.documents import Document


def _heading_level(para) -> int | None:
    """识别 Heading 1/2（兼容中英文样式名与 style_id）。"""
    style = para.style
    name = (style.name or "").lower()
    sid = (style.style_id or "").lower()
    if name.startswith("heading 1") or name.startswith("标题 1") or sid == "heading1":
        return 1
    if name.startswith("heading 2") or name.startswith("标题 2") or sid == "heading2":
        return 2
    return None


def parse_docx_to_sections(file_path: str) -> list[dict[str, Any]]:
    """按 Heading 章节拆页；空文档返回 []。返回 [{"page", "title", "text"}]。"""
    doc = DocxDocument(file_path)
    parts: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        level = _heading_level(para)
        if level == 1:
            parts.append(f"\n# {text}\n")
        elif level == 2:
            parts.append(f"\n## {text}\n")
        else:
            parts.append(text)

    linear = "\n".join(parts).strip()
    if not linear:
        return []

    pieces = re.split(r"(?m)^(?=## .+$)", linear)
    pieces = [p.strip() for p in pieces if p.strip()]
    if len(pieces) <= 1:
        return [{"page": 1, "title": "", "text": linear}]

    sections: list[dict[str, Any]] = []
    for i, block in enumerate(pieces, start=1):
        first_line = block.split("\n", 1)[0].strip()
        sec_title = ""
        if first_line.startswith("## "):
            sec_title = first_line[3:].strip()
        elif i == 1 and first_line.startswith("# ") and not first_line.startswith("##"):
            sec_title = first_line[2:].strip()
        prefix = f"[章节: {sec_title}]\n\n" if sec_title else ""
        sections.append({"page": i, "title": sec_title, "text": prefix + block})
    return sections


def load_docx_for_document_loader(file_path: str, filename: str) -> list[Document]:
    """供 DocumentLoader 使用：返回 LangChain Document 列表，metadata.page 为章节序号。"""
    docs: list[Document] = []
    for sec in parse_docx_to_sections(file_path):
        docs.append(
            Document(
                page_content=sec["text"],
                metadata={
                    "page": sec["page"],
                    "source": filename,
                    "section_title": sec["title"],
                },
            )
        )
    return docs
