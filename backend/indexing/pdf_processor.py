"""PDF 结构化提取：基于 MinerU pipeline 解析，按 Markdown 标题拆成多「页」供三层分块复用。

MinerU 输出 Markdown（含 #/## 标题层级），复用与 docx_processor 相同的章节拆页逻辑。
模型复用 modelscope 本地缓存（OpenDataLab/MinerU2.5-Pro-2605-1.2B + PDF-Extract-Kit-1.0）。
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from typing import Any

from langchain_core.documents import Document


def _split_markdown_sections(md_text: str) -> list[dict[str, Any]]:
    """按 Markdown 二级标题拆页；与 docx_processor 的章节拆页逻辑同构。"""
    linear = md_text.strip()
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


def parse_pdf_with_mineru(file_path: str) -> list[dict[str, Any]]:
    """使用 MinerU 本地 pipeline 解析 PDF，返回 [{"page", "title", "text"}]。"""
    from mineru.cli.common import do_parse

    filename = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = tempfile.mkdtemp(prefix="mineru_")
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()

    try:
        do_parse(
            output_dir=output_dir,
            pdf_file_names=[filename],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["ch"],
            backend="pipeline",
            parse_method="auto",
        )

        md_path = os.path.join(output_dir, filename, "auto", f"{filename}.md")
        if not os.path.exists(md_path):
            raise FileNotFoundError(f"MinerU 输出文件不存在: {md_path}")

        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)

    if not md_text.strip():
        raise ValueError("PDF 解析结果为空")

    return _split_markdown_sections(md_text)


def load_pdf_for_document_loader(file_path: str, filename: str) -> list[Document]:
    """供 DocumentLoader 使用：返回 LangChain Document 列表，metadata.page 为章节序号。"""
    docs: list[Document] = []
    for sec in parse_pdf_with_mineru(file_path):
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
