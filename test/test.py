import os
import json
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError, NoCredentialsError

load_dotenv()

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Runtime client (for invoking models)
client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION", REGION),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


def test_embeddings():
    print("Test 1: Amazon Titan Embeddings")
    response = client.invoke_model(
        modelId=os.getenv("BEDROCK_EMBED_MODEL"),
        body=json.dumps({
            "inputText": "Hello Amazon Bedrock",
            "dimensions": 1024,
            "normalize": True,
        }),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    embedding = result["embedding"]
    print(f"  PASSED - got embedding with {len(embedding)} dimensions")
    print(f"  First 5 values: {embedding[:5]}")


#amazon bedrock test for text generation
def test_text_generation():
    print("Test 2: Text generation (Converse API)")
    model_id = os.getenv("BEDROCK_CHAT_MODEL")
    if not model_id:
        print("  SKIPPED - set BEDROCK_CHAT_MODEL in .env to run this test")
        return
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": "Say hello in one short sentence."}]}],
        inferenceConfig={"maxTokens": 100, "temperature": 0.5},
    )
    text = response["output"]["message"]["content"][0]["text"]
    print(f"  PASSED - model replied: {text}")


def run(test_fn):
    try:
        test_fn()
    except NoCredentialsError:
        print("  FAILED - AWS credentials not found. Check your .env file.")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]
        print(f"  FAILED - {code}: {msg}")
        if code == "AccessDeniedException":
            print("  Hint: enable model access in the Bedrock console, and check IAM permissions.")
        elif code == "ValidationException":
            print("  Hint: check the model ID and request body format.")
        elif code == "ResourceNotFoundException":
            print("  Hint: model not available in this region.")
    except Exception as e:
        print(f"  FAILED - {type(e).__name__}: {e}")


if __name__ == "__main__":
    print(f"Region: {REGION}\n")
    run(test_embeddings)
    print()
    run(test_text_generation)