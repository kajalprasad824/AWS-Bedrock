import boto3
import json

prompt_data = """
Act as Shakspeare and write a poem about machine learning
"""

#Create bedrock runtime client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

#Model ID
model_id="nvidia.nemotron-super-3-120b"

# Request Payload
payload={
    "messages": [
        {
            "role":"user",
            "content":prompt_data
        }
    ],
    "max_tokens":512,
    "temperature":0.5,
    "top_p":0.9
    }

# Invoke the model
response = bedrock.invoke_model(
    body=json.dumps(payload),
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

# Read the response
response_body = json.loads(response["body"].read())

# Extract and print the generated text
response_text = response_body["choices"][0]["message"]["content"]

print(response_text)

