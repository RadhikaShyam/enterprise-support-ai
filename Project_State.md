# Enterprise Support AI — Project State

## Project Goal

Build a production-oriented enterprise support AI platform combining:

- RAG
- parameter-efficient fine-tuning
- LoRA / QLoRA
- preference optimization
- tool calling
- agentic workflows
- evaluation
- FastAPI
- PostgreSQL + pgvector
- Docker
- deployment

## Hardware Constraints

- GPU: NVIDIA RTX 3050
- VRAM: 6 GB
- Development must work on local hardware.
- Avoid large-model full fine-tuning.
- Prefer small open-source models and quantization.
- No paid API dependency.

## Current Phase

Phase 1 — Backend foundation

## Completed

- Python 3.11 virtual environment
- Git installed
- Docker installed
- Project directory created

## Current Task

Create FastAPI backend and verify the application runs.

## Next

1. FastAPI application
2. PostgreSQL
3. Database models
4. Document upload
5. Document parsing
6. Chunking
7. Embeddings
8. pgvector retrieval
9. RAG
10. Fine-tuning pipeline

## Important Decisions

- Python 3.11
- FastAPI backend
- PostgreSQL database
- pgvector for vector search
- Local/open-source models
- PEFT/LoRA/QLoRA for fine-tuning
- Git for version control

## Current Blockers

None