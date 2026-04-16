import boto3
from botocore.config import Config
import json
 
client_args = {
    "connect_timeout": 10,
    "read_timeout": 300,
    "retries" : {
            'max_attempts': 10,  # Maximum retry attempts
            'mode': 'standard'   # Retry mode: 'legacy', 'standard', or 'adaptive'
        }
}
 
 
import boto3
 
modelId = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
 
client = boto3.client(
    "bedrock-runtime",
    config = Config(**client_args),
    region_name= "us-east-1"
)
 
sys_prompt = [{"text": "You are sweet and good"}]
 
messages = [{"role": "user", 
             "content": [{"text": "Hi"}]}]
 
inferenceConfig = {
        'maxTokens': 4096,
        'temperature': 0
    }
 
 
response = client.converse(messages=messages, 
                            system=sys_prompt, 
                            modelId=modelId,
                            inferenceConfig=inferenceConfig)
print(json.dumps(response, indent=4))