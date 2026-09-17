# Contributing to Production RAG

Thank you for your interest in contributing to **Production RAG**! We are building the world's most modular, developer-agnostic, enterprise-grade RAG engine, and we welcome contributions from developers worldwide.

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please treat all contributors with respect and kindness.

---

## Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/<your-username>/RAG-production.git
cd RAG-production
```

### 2. Local Environment Setup

We recommend Python 3.10, 3.11, or 3.12:

```bash
# Create virtual environment
python -m venv .venv

# Activate on Linux/macOS
source .venv/bin/activate

# Activate on Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Install in editable mode with development dependencies
pip install -e ".[all,dev]"
```

### 3. Start Local Postgres (pgvector)

```bash
docker compose up -d postgres
```

### 4. Verify System Diagnostics

```bash
rag doctor
# or: python main.py doctor
```

---

## Architectural Guidelines

### Adding a New LLM Provider Adapter

All LLM generators implement `BaseGenerator` in `pipeline/generator.py`:

```python
from pipeline.generator import BaseGenerator, format_context_and_citations, filter_active_citations

class MyCustomGenerator(BaseGenerator):
    def __init__(self, api_key: str = None, model_name: str = "custom-model"):
        self.model_name = model_name
        # Initialize client

    def generate(self, query: str, chunks: list[dict]) -> tuple[str, list[dict]]:
        if not chunks:
            return "No documents found.", []
        context_str, citations = format_context_and_citations(chunks)
        
        # Call model
        answer = "..." 
        return answer, filter_active_citations(answer, citations)
```

Register your generator in `get_generator()` factory and add any necessary configuration fields to `pipeline/config.py`.

### Adding a New Embedder

Implement `BaseEmbedder` in `pipeline/embedder.py`:
- Provide `embed_query(text: str) -> list[float]`
- Provide `embed_documents(texts: list[str]) -> list[list[float]]`
- Provide `dimension` property.

---

## Testing & Quality Assurance

Run the test suite before submitting any Pull Request:

```bash
# Run unit & integration tests
pytest test_pipeline.py test_production_features.py test_tenancy.py -v

# Run linting check
ruff check .
```

---

## Submitting a Pull Request

1. Create a descriptive feature branch:
   ```bash
   git checkout -b feat/add-qdrant-vectorstore
   ```
2. Commit your changes with clear commit messages following Conventional Commits:
   ```bash
   git commit -m "feat(providers): add Qdrant vector store adapter"
   ```
3. Push to your fork and open a Pull Request against the `main` branch.
4. Ensure all CI checks pass.

---

## Need Help?

- Join the discussion in [GitHub Issues](https://github.com/njNagaraj/RAG-production/issues).
- Tag your issue with `good first issue` or `question`.
