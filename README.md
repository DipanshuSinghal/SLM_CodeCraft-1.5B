# CodeCraft-1.5B

CodeCraft-1.5B is a localized, containerized Small Language Model (SLM) for enterprise Python code synthesis and automated test generation.

## Features
- **QLoRA Fine-Tuning**: Efficient 4-bit adaptation built on `Qwen2.5-Coder-1.5B`.
- **FastAPI Endpoint**: Sub-100ms per-token generation interface.
- **Docker-Ready**: Optimized multi-stage build setup.
- **Cloud Native**: Pre-configured for AWS ECS / ECR deployment.

## Execution Workflow

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt