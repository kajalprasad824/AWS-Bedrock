
import json
import logging
import boto3
import botocore.config
from datetime import datetime, timezone


# --------------------------------------------------
# Configuration
# --------------------------------------------------

REGION_NAME = "us-east-1"
MODEL_ID = "amazon.nova-pro-v1:0"
S3_BUCKET = "bedrockbloggenerationdemo"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=REGION_NAME,
    config=botocore.config.Config(
        read_timeout=300,
        retries={"max_attempts": 3}
    )
)

s3 = boto3.client("s3", region_name=REGION_NAME)


# --------------------------------------------------
# Generate blog using Amazon Bedrock Nova Pro
# --------------------------------------------------

def blog_generate_using_bedrock(blogtopic: str) -> str:
    prompt = (
        f"Write a well-structured blog of approximately 200 words "
        f"on the following topic: {blogtopic}. "
        "Include an introduction, useful details, and a conclusion."
    )

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": prompt}
                ]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 512,
            "temperature": 0.5,
            "topP": 0.9
        }
    }

    try:
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        response_data = json.loads(response["body"].read())

        blog_details = response_data["output"]["message"]["content"]

        # Combine any returned text blocks
        blog_text = "\n".join(
            item["text"]
            for item in blog_details
            if "text" in item
        )

        if not blog_text.strip():
            raise ValueError("Bedrock returned an empty blog.")

        return blog_text

    except Exception:
        logger.exception("Error generating blog with Bedrock")
        raise


# --------------------------------------------------
# Save generated blog to Amazon S3
# --------------------------------------------------

def save_blog_details_s3(
    s3_key: str,
    s3_bucket: str,
    blog_content: str
) -> None:

    s3.put_object(
        Bucket=s3_bucket,
        Key=s3_key,
        Body=blog_content.encode("utf-8"),
        ContentType="text/plain; charset=utf-8"
    )

    logger.info("Blog saved successfully to s3://%s/%s",
                s3_bucket, s3_key)


# --------------------------------------------------
# AWS Lambda handler
# --------------------------------------------------

def lambda_handler(event, context):
    try:
        # Support API Gateway requests and direct Lambda test events
        if "body" in event:
            request_body = event["body"]

            if event.get("isBase64Encoded"):
                import base64
                request_body = base64.b64decode(
                    request_body
                ).decode("utf-8")

            if isinstance(request_body, str):
                request_body = json.loads(request_body)
        else:
            request_body = event

        blogtopic = request_body.get("blog", "").strip()

        if not blogtopic:
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": json.dumps({
                    "error": "Please provide a blog topic in the 'blog' field."
                })
            }

        # Generate the blog
        generate_blog = blog_generate_using_bedrock(
            blogtopic=blogtopic
        )

        # Create a unique S3 object key
        current_time = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%S%fZ")

        s3_key = f"blog-output/{current_time}.txt"

        # Save the generated content to S3
        save_blog_details_s3(
            s3_key=s3_key,
            s3_bucket=S3_BUCKET,
            blog_content=generate_blog
        )

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "message": "Blog generated and saved successfully.",
                "blog": generate_blog,
                "s3_bucket": S3_BUCKET,
                "s3_key": s3_key
            })
        }

    except Exception:
        logger.exception("Lambda handler failed")

        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "error": "Failed to generate or save the blog."
            })
        }
