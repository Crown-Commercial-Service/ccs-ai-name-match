# ccs-ai-name-matcher

A lightweight **FastAPI-based microservice** for **LLM-powered name matching**, designed to match an input string (e.g. buyer or supplier name) to the most appropriate option from a list of candidate strings.

This service is intended to be reusable across CCS projects and supports both:
- **Azure OpenAI (via LangChain)** for production usage
- **Mock LLM logic** for local development and testing without external dependencies

## Overview

The service exposes a simple HTTP API that:
1. Accepts an `input_string`
2. Accepts a list of `candidates`
3. Uses an LLM to determine the best match (or returns `None`)

The core logic is intentionally minimal and modular so it can:
- Run as a standalone microservice
- Be called by dashboards, batch jobs or data pipelines
- Switch between mock and Azure OpenAI with minimal changes

## Project Structure

```
ccs-ai-name-matcher/
│
├── app/
│ ├── main.py
│ ├── config.py
│ ├── model_factory.py
│ ├── langchain_matcher.py
│ └── services/
│ └── mock_langchain_model.py # Mock LLM for local/dev use
│
├── prompts/
│ └── buyer_match_v*.txt # Prompt templates (versioned)
│
├── tests/
│ ├── conftest.py # Test fixtures
│ ├── test_api.py # API endpoint tests
│ ├── test_mock_model.py # Mock model tests
│ └── test_prompts.py # Prompt loading tests
│
├── env.example.txt
├── pytest.ini
├── requirements.txt
├── app.py
└── README.md
```

## Installation

### Python Environment

1. Create and activate a virtual environment
```bash
python -m venv .venv
```
2. Update pip
```bash
pip install --upgrade pip
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

### Environment Variables

To enable Azure OpenAI instead of the mock LLM, first copy the example `env.example.txt` file into a new `.env` file

```bash
cp env.example.txt .env
```

Then update the `.env` file with valid Azure OpenAI credentials:

```env
AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/
AZURE_OPENAI_KEY=<your-api-key>
AZURE_OPENAI_DEPLOYMENT_NAME=<your-deployment-name>
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

## Running the Service

To run this service locally, you can either use uvicorn:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or run via Python entrypoint (which still uses uvicorn under the hood):

```bash
python app.py
```

## API Endpoints

Once the service is running (see above), there are 2 endpoints that you can call. For demo purposes, these can be called by issuing `curl` commands through the terminal, as shown below.

### Health Check

GET /health: used to verify the service is running.

```bash
curl http://127.0.0.1:8000/health
```

### Name Matching

GET /match: used to send a name of interest and a string of potential matches
```bash
curl "http://127.0.0.1:8000/match?input_string=Home%20Ofice&candidates=Home%20Office&candidates=HMRC"
```

## Tests

The test suite includes 24 automated tests which covers:
* API endpoints (health check, GET/POST /match)
* Input validation (empty strings, missing fields)
* Matching behavior (exact match, fuzzy match, no match)
* Case-insensitive matching
* Custom prompt file loading
* Mock model logic
* Real-world UK procurement scenarios

To run the tests:

```bash
pytest
```

To run tests with verbose output:

```bash
pytest -v
```

To run a specific test file:

```bash
pytest tests/test_api.py
```
