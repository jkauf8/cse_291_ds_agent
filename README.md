# AWS Bedrock LLM Setup

This directory contains code for the ML-based Network Intrusion Detection System project.

## AWS Bedrock Setup

### Prerequisites
1. AWS Account with Bedrock access
2. Python 3.8+
3. AWS credentials (Access Key ID and Secret Access Key)

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure AWS credentials:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your AWS credentials:
     ```
     AWS_ACCESS_KEY_ID=your_actual_access_key
     AWS_SECRET_ACCESS_KEY=your_actual_secret_key
     AWS_DEFAULT_REGION=us-east-1
     ```

### Enabling Bedrock Models

Before using Bedrock, you need to enable model access in AWS Console:

1. Go to AWS Console → Bedrock → Model access
2. Request access to the models you want to use:
   - Claude 3 Haiku (recommended for cost-effective testing)
   - Claude 3 Sonnet
   - Claude 3.5 Sonnet
   - Amazon Titan Text models

### Usage

Run the example script:
```bash
python bedrock_client.py
```

Or use in your own code:
```python
from bedrock_client import BedrockClient

# Initialize client
client = BedrockClient()

# Send a prompt
response = client.invoke(
    prompt="What is machine learning?",
    model='claude-3-haiku',
    max_tokens=1024,
    temperature=0.7
)

print(response)
```

### Available Models

- `claude-3-haiku` - Fast and cost-effective (recommended for testing)
- `claude-3-sonnet` - Balanced performance
- `claude-3-5-sonnet` - Most capable
- `titan-text-express` - Amazon's model
- `titan-text-lite` - Lightweight Amazon model

### Troubleshooting

**Error: "Could not load credentials"**
- Make sure your `.env` file exists and contains valid credentials

**Error: "AccessDeniedException"**
- Enable model access in AWS Bedrock console
- Check your IAM permissions include `bedrock:InvokeModel`

**Error: "ValidationException: The provided model identifier is invalid"**
- Make sure you've requested access to the model in Bedrock console
- Check the region supports the model you're trying to use
