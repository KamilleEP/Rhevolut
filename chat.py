import json
import os
import boto3
import xml.etree.ElementTree as ET
from langchain.memory import DynamoDBChatMessageHistory


AWS_REGION = os.getenv("AWS_REGION")
DYNAMO_TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME")

client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(DYNAMO_TABLE_NAME)

VALID_MODEL_IDS = [
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
]


def parse_event_body(event):
    """Parse and return the JSON body of the event."""
    try:
        return json.loads(event.get("body", "{}")), None
    except json.JSONDecodeError as e:
        return None, {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid JSON in request body"}),
        }


def validate_request(body):
    """Validate the request body for required fields."""
    question = body.get("question")
    model_id = body.get("modelId")
    prompt = body.get("prompt")
    temperature = body.get("temperature", 0.1)
    top_k = body.get("top_k", 1)
    top_p = body.get("top_p", 0.1)
    max_tokens = body.get("max_tokens", 1000)

    required_fields = ["question", "modelId"]
    missing_fields = [field for field in required_fields if not body.get(field)]
    if missing_fields:
        return None, {
            "statusCode": 400,
            "body": json.dumps(
                {"error": f"Missing required fields: {', '.join(missing_fields)}"}
            ),
        }

    if model_id not in VALID_MODEL_IDS:
        return None, {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid modelId in request body"}),
        }

    return (question, model_id, prompt, top_k, top_p, max_tokens, temperature), None

def retrive_documents(input_configuration):
    """Call the AWS service to retrieve documents based on the input configuration."""
    try:
        response = client.retrieve(**input_configuration)
        return response, None
    except Exception as e:
        print(e)
        return None, {
            "statusCode": 500,
            "body": json.dumps(
                {"error": "Server side error: please check function logs"}
            ),
        }


def generate_response(input_configuration):
    """Call the AWS service to generate a response based on the input configuration."""
    try:
        response = bedrock_runtime.invoke_model(**input_configuration)
        return response, None
    except Exception as e:
        print(e)
        return None, {
            "statusCode": 500,
            "body": json.dumps(
                {"error": "Server side error: please check function logs"}
            ),
        }


def get_question_topics(question, model_id):
    """Get the topics of the question."""

    prompt = f"""
        <pergunta>
            {question}
        <pergunta>

        retorne uma lista de todos os tópicos, palavras chaves e sinônimos comuns no contexto da pergunta que esta <pergunta> deseja saber

        formate a saida em um xml, retorne somente o xml e nada mais

        <lista>
            <topico>[tópicos e palavras chaves da pergunta]</topico>
        </lista>
    """

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 500,
        }
    )

    invoke_configuration = {
        "body": body,
        "modelId": model_id,
        "accept": "application/json",
        "contentType": "application/json",
    }

    response = bedrock_runtime.invoke_model(**invoke_configuration)

    response_body = json.loads(response.get("body").read().decode("utf-8"))

    texts = []
    for content in response_body.get("content", []):
        texts.append(content.get("text", {}))

    root = ET.fromstring(texts[0])

    topics = []

    for topic in root.findall("topico"):
        topics.append(topic.text)

    topics.append(question)

    print(f"[DEBUG][get_question_topics] topics: {topics}")

    return topics

def create_invoke_configuration(question, model_id, top_k, top_p, max_tokens, temperature, custom_prompt=""):
    print(f"[DEBUG][create_invoke_configuration] custom_prompt: {custom_prompt}")
    texts = []

    """Create the input configuration for the AWS service call."""
    prompt = f"""
    <pergunta>
        {question}
    </pergunta>

    {custom_prompt}
    """

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": int(max_tokens),
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": int(temperature),
            "top_p": int(top_p),
            "top_k": int(top_k),
        }
    )

    invoke_configuration = {
        "body": body,
        "modelId": model_id,
        "accept": "application/json",
        "contentType": "application/json",
    }

    print('invoke_configuration', invoke_configuration)

    return invoke_configuration


def handler(event, context):
    body, error = parse_event_body(event)
    if error:
        return error

    request_data, error = validate_request(body)
    if error:
        return error

    question, model_id, prompt, top_k, top_p, max_tokens, temperature = request_data

    print(
        f"[DEBUG][handler] question: {question}, model_id: {model_id}, top_k: {top_k}, top_p: {top_p}, max_tokens: {max_tokens}, temperature: {temperature}"
    )

    invoke_configuration = create_invoke_configuration(
        question, model_id, top_k, top_p, max_tokens, temperature, prompt
    )

    response, error = generate_response(invoke_configuration)
    if error:
        return error

    response_body = json.loads(response.get("body").read().decode("utf-8"))

    texts = []
    for content in response_body.get("content", []):
        texts.append(content.get("text", {}))

    output = " ".join(texts)

    print(f"[DEBUG][handler] output: {output}")

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "response": output
            }
        ),
        "headers": {"Access-Control-Allow-Origin": "*"},
    }
