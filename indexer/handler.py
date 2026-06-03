import os
import json
import boto3
import tempfile
from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.typing import LambdaContext

from pdf_parser import parse_pdf
from chunker import chunk_pages
from embedder import embed_and_index

logger = Logger(service="databricks-cert-rag-indexer")

s3_client = boto3.client("s3", region_name="us-east-1")


@logger.inject_lambda_context(log_event=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    logger.info("Indexer Lambda invoked", extra={
        "record_count": len(event.get("Records", [])),
    })

    results = []

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        file_size = record["s3"]["object"].get("size", 0)

        logger.info("Processing S3 record", extra={
            "bucket": bucket,
            "s3_key": key,
            "file_size_bytes": file_size,
        })

        try:
            result = process_pdf(bucket, key)
            results.append({
                "s3_key": key,
                "status": "success",
                **result,
            })
            logger.info("Record processed successfully", extra={
                "s3_key": key,
                "pages": result["pages"],
                "chunks": result["chunks"],
                "indexed": result["indexed"],
                "failed": result["failed"],
            })

        except Exception as e:
            logger.error("Failed to process record", extra={
                "s3_key": key,
                "error": str(e),
            })
            results.append({
                "s3_key": key,
                "status": "failed",
                "error": str(e),
            })

    logger.info("Indexer Lambda complete", extra={
        "total_records": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
    })

    return {
        "statusCode": 200,
        "body": json.dumps({"results": results}),
    }


def process_pdf(bucket: str, key: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        local_path = tmp.name

    try:
        logger.info("Downloading PDF from S3", extra={
            "bucket": bucket,
            "s3_key": key,
            "local_path": local_path,
        })
        s3_client.download_file(bucket, key, local_path)
        logger.info("PDF downloaded successfully", extra={"s3_key": key})

        logger.info("Parsing PDF", extra={"s3_key": key})
        pages = parse_pdf(local_path, key)
        logger.info("Parsing complete", extra={
            "s3_key": key,
            "pages": len(pages),
        })

        logger.info("Chunking pages", extra={"s3_key": key})
        chunks = chunk_pages(pages)
        logger.info("Chunking complete", extra={
            "s3_key": key,
            "chunks": len(chunks),
        })

        logger.info("Embedding and indexing", extra={"s3_key": key})
        index_result = embed_and_index(chunks)
        logger.info("Embedding and indexing complete", extra={
            "s3_key": key,
            "indexed": index_result["indexed"],
            "failed": index_result["failed"],
        })

        return {
            "pages": len(pages),
            "chunks": len(chunks),
            "indexed": index_result["indexed"],
            "failed": index_result["failed"],
        }

    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.info("Temp file cleaned up", extra={"local_path": local_path})