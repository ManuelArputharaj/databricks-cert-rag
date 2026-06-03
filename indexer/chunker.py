import tiktoken
import uuid
from aws_lambda_powertools import Logger

logger = Logger(service="databricks-cert-rag-indexer")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
ENCODING_MODEL = "text-embedding-3-small"


def get_encoder():
    return tiktoken.encoding_for_model(ENCODING_MODEL)


def chunk_pages(pages: list[dict]) -> list[dict]:
    encoder = get_encoder()
    all_chunks = []

    logger.info("Starting chunking", extra={"total_pages": len(pages)})

    for page in pages:
        text = page["text"]
        tokens = encoder.encode(text)

        if len(tokens) == 0:
            logger.debug("Skipping empty page text", extra={
                "exam_name": page["exam_name"],
                "page_number": page["page_number"]
            })
            continue

        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + CHUNK_SIZE, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = encoder.decode(chunk_tokens)

            if chunk_text.strip():
                chunk = {
                    "chunk_id": str(uuid.uuid4()),
                    "exam_name": page["exam_name"],
                    "section": page["section"],
                    "page_number": page["page_number"],
                    "s3_key": page["s3_key"],
                    "chunk_text": chunk_text.strip(),
                    "token_count": len(chunk_tokens),
                    "chunk_index": chunk_index,
                }
                all_chunks.append(chunk)
                logger.debug("Chunk created", extra={
                    "chunk_id": chunk["chunk_id"],
                    "exam_name": page["exam_name"],
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                    "token_count": len(chunk_tokens),
                })
                chunk_index += 1

            if end == len(tokens):
                break

            start = end - CHUNK_OVERLAP

    logger.info("Chunking complete", extra={
        "total_pages": len(pages),
        "total_chunks": len(all_chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    })

    return all_chunks