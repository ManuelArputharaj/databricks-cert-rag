import fitz
import re
from aws_lambda_powertools import Logger

logger = Logger(service="databricks-cert-rag-indexer")

EXAM_NAME_MAP = {
    "data-analyst-associate": "Databricks Certified Data Analyst Associate",
    "data-engineer-associate": "Databricks Certified Data Engineer Associate",
    "data-engineer-professional": "Databricks Certified Data Engineer Professional",
    "genai-engineer-associate": "Databricks Certified Generative AI Engineer Associate",
    "ml-associate": "Databricks Certified Machine Learning Associate",
    "ml-professional": "Databricks Certified Machine Learning Professional",
}


def resolve_exam_name(s3_key: str) -> str:
    key_lower = s3_key.lower()
    for slug, name in EXAM_NAME_MAP.items():
        if slug in key_lower:
            return name
    filename = s3_key.split("/")[-1].replace(".pdf", "").replace("_", "-")
    logger.warning("Could not match exam name from S3 key, using filename", extra={"s3_key": s3_key, "fallback": filename})
    return filename


def parse_pdf(local_path: str, s3_key: str) -> list[dict]:
    exam_name = resolve_exam_name(s3_key)
    logger.info("Starting PDF parse", extra={"s3_key": s3_key, "exam_name": exam_name, "local_path": local_path})

    doc = fitz.open(local_path)
    pages = []
    current_section = "General"

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        if not text:
            logger.debug("Skipping empty page", extra={"page_number": page_num + 1})
            continue

        section = extract_section(text) or current_section
        current_section = section

        pages.append({
            "exam_name": exam_name,
            "section": section,
            "page_number": page_num + 1,
            "text": clean_text(text),
            "s3_key": s3_key,
        })

    doc.close()
    logger.info("PDF parse complete", extra={"s3_key": s3_key, "total_pages": len(pages)})
    return pages


def extract_section(text: str) -> str | None:
    section_patterns = [
        r"^Section\s+\d+[:\s]+(.+)$",
        r"^(Exam Outline|About the Exam|Audience Description|Sample Questions|Recommended Training)$",
    ]
    for line in text.split("\n")[:5]:
        line = line.strip()
        for pattern in section_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                matched = match.group(1) if match.lastindex else line
                logger.debug("Section detected", extra={"section": matched})
                return matched
    return None


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()