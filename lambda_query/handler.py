import json
from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.typing import LambdaContext

from search import search
from router import route

logger = Logger(service="databricks-cert-rag-query")


@logger.inject_lambda_context(log_event=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    logger.info("Query Lambda invoked")

    try:
        body = json.loads(event.get("body", "{}"))
        query = body.get("query", "").strip()
        exam_name = body.get("exam_name", None)

        logger.info("Request parsed", extra={
            "query": query,
            "exam_filter": exam_name,
        })

        if not query:
            logger.warning("Empty query received")
            return build_response(400, {"error": "query field is required"})

        logger.info("Executing search", extra={"query": query})
        search_results = search(query=query, exam_name=exam_name)

        logger.info("Executing route", extra={
            "query": query,
            "results_count": len(search_results),
        })
        result = route(query=query, search_results=search_results)

        logger.info("Query Lambda complete", extra={
            "query": query,
            "tier": result["tier"],
            "tier_label": result["tier_label"],
            "score": result["score"],
            "sources_count": len(result.get("sources", [])),
        })

        return build_response(200, {
            "query": query,
            "exam_filter": exam_name,
            **result,
        })

    except Exception as e:
        logger.error("Unhandled error in Query Lambda", extra={
            "error": str(e),
        })
        return build_response(500, {
            "error": "Internal server error",
            "detail": str(e),
        })


def build_response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "POST,OPTIONS",
        },
        "body": json.dumps(body),
    }