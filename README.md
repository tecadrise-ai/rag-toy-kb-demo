# Aurora Toy RAG (public demo)

**Repository:** [github.com/tecadrise-ai/rag-toy-kb-demo](https://github.com/tecadrise-ai/rag-toy-kb-demo)

Small **retrieval augmented generation** demo for learning and portfolio use. The knowledge base is **100% fake**: a helpdesk for a made up company, **Aurora Bean Roasters** (B2B coffee wholesaler). It is **not** related to any client project.

**What it does**

1. **Stage 1:** Vector search over short **policy** excerpts (shipping, returns, billing, and so on).
2. **Stage 2:** Vector search over **playbooks** (troubleshooting scripts), with the user question and the top policies concatenated to steer retrieval.

Embeddings run **locally** with `sentence-transformers` (`all-MiniLM-L6-v2`). Vectors are stored in an on disk **Chroma** database under `.chroma_data/`.

**Optional LLM step:** If you set `OPENAI_API_KEY` in a local `.env` file, you can add `--llm` to draft a short answer that is instructed to use **only** the retrieved chunks. No API keys belong in the repository.

## Quick start

Requirements: **Python 3.10+**

```text
cd rag-toy-kb-demo
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/ingest.py
python main.py
```

**Single question**

```text
python main.py "the pallet arrived but we did not order a lift gate" --debug
```

**With OpenAI (optional)**

```text
copy .env.example .env
REM edit .env and set OPENAI_API_KEY
python main.py "customer wants to return unopened bags from last week" --llm
```

## Data layout

| Path | Content |
|------|---------|
| `data/policies.json` | Policy snippets (synthetic) |
| `data/playbooks.json` | Playbooks (synthetic) |
| `.chroma_data/` | Generated vector store (gitignored) |

## Security and publishing

* Copy `.env.example` to `.env` and keep `.env` **local** and out of git.
* Do not commit real API keys, customer data, or internal URLs.
* The sample JSON is fictional; replace with your own content if you fork the project.

## Publishing to GitHub

Create a new empty repository on GitHub, then either:

* Copy the `rag-toy-kb-demo` folder into a fresh clone, or
* From this monorepo, copy only the `rag-toy-kb-demo` tree into a new project directory.

In the project root, run `git init`, add files, commit, add the `origin` remote, and push. Do not commit `.env`, `.venv/`, or `.chroma_data/`.

## License

You may use this demo freely for learning and as a template. Add your own `LICENSE` file if you need a specific license for your GitHub org.
