import os
import json
import boto3
from aws_lambda_powertools import Logger

logger = Logger(service="databricks-cert-rag-query")

BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
CLAUDE_MODEL_ID = "anthropic.claude-sonnet-4-6"
TIER1_THRESHOLD = 0.85
TIER2_THRESHOLD = 0.60


def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


def invoke_claude(prompt: str) -> tuple[str, int, int]:
    bedrock_client = get_bedrock_client()

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    })

    response = bedrock_client.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    result = json.loads(response["body"].read())
    answer = result["content"][0]["text"]
    input_tokens = result["usage"]["input_tokens"]
    output_tokens = result["usage"]["output_tokens"]

    return answer, input_tokens, output_tokens


def route(query: str, search_results: list[dict]) -> dict:
    logger.info("Starting route decision", extra={
        "query": query,
        "results_count": len(search_results),
    })

    if not search_results:
        logger.info("No search results found — routing to Tier 3")
        return tier3_claude_only(query)

    top_score = search_results[0]["score"]
    sources = build_sources(search_results)

    logger.info("Score evaluated", extra={
        "top_score": round(top_score, 4),
        "tier1_threshold": TIER1_THRESHOLD,
        "tier2_threshold": TIER2_THRESHOLD,
    })

    if top_score >= TIER1_THRESHOLD:
        logger.info("Routing to Tier 1 — direct answer from OpenSearch", extra={
            "top_score": round(top_score, 4),
            "exam_name": search_results[0]["exam_name"],
            "section": search_results[0]["section"],
            "page_number": search_results[0]["page_number"],
        })
        return {
            "tier": 1,
            "tier_label": "Direct from exam guide",
            "answer": search_results[0]["chunk_text"],
            "score": round(top_score, 4),
            "sources": sources,
        }

    elif top_score >= TIER2_THRESHOLD:
        logger.info("Routing to Tier 2 — augmenting with Claude", extra={
            "top_score": round(top_score, 4),
            "context_chunks": len(search_results[:3]),
        })
        answer = tier2_augment(query, search_results)
        return {
            "tier": 2,
            "tier_label": "Augmented by Claude",
            "answer": answer,
            "score": round(top_score, 4),
            "sources": sources,
        }

    else:
        logger.info("Routing to Tier 3 — Claude direct answer", extra={
            "top_score": round(top_score, 4),
        })
        return tier3_claude_only(query)


def tier2_augment(query: str, search_results: list[dict]) -> str:
    context = "\n\n---\n\n".join([
        f"[{r['exam_name']} | {r['section']} | Page {r['page_number']}]\n{r['chunk_text']}"
        for r in search_results[:3]
    ])

    prompt = f"""You are a Databricks certification expert assistant.

Use the following retrieved context from the official Databricks exam guides to answer the question.
If the context is partially relevant, use it and supplement with your knowledge.
Be concise and accurate.

Context:
{context}

Question: {query}

Answer:"""

    logger.info("Calling Claude via Bedrock for Tier 2 augmentation", extra={
        "query": query,
        "context_length": len(context),
        "model": CLAUDE_MODEL_ID,
    })

    answer, input_tokens, output_tokens = invoke_claude(prompt)

    logger.info("Tier 2 Claude response received", extra={
        "query": query,
        "response_length": len(answer),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    })

    return answer


def tier3_claude_only(query: str) -> dict:
    prompt = f"""You are a Databricks certification expert assistant.
Answer the following question about Databricks certifications accurately and concisely.

Question: {query}

Answer:"""

    logger.info("Calling Claude via Bedrock for Tier 3 direct answer", extra={
        "query": query,
        "model": CLAUDE_MODEL_ID,
    })

    answer, input_tokens, output_tokens = invoke_claude(prompt)

    logger.info("Tier 3 Claude response received", extra={
        "query": query,
        "response_length": len(answer),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    })

    return {
        "tier": 3,
        "tier_label": "Claude direct answer",
        "answer": answer,
        "score": 0.0,
        "sources": [],
    }


def build_sources(results: list[dict]) -> list[dict]:
    return [
        {
            "exam_name":   r["exam_name"],
            "section":     r["section"],
            "page_number": r["page_number"],
            "score":       round(r["score"], 4),
        }
        for r in results[:3]
    ]