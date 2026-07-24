# Paul Agent — Tool Routing & Parameter Extraction Benchmark

Benchmark pipeline for an LLM agent that routes natural-language requests to the right
[Slicer Automated Dental Tools](https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools) CLI
(ALI, AMASSS, ASO, AReg, CLI-C, MedX, …) and extracts the correct CLI parameters from the request.

The agent pipeline has three stages, evaluated both independently and end-to-end:

1. **Retrieval (RAG)** — narrow the full tool catalog down to the most relevant candidates for a
   query (BM25, vector, cross-encoder reranker, or hybrid).
2. **Routing** — an LLM picks the single best tool from the retrieved candidates.
3. **Parameter extraction** — an LLM (or agent) extracts the tool's CLI parameters from the
   natural-language request.

## Repository layout

```
RAG/            Stage 1 only: retrieval-mode comparison (BM25 / Vector / cross-encoder / hybrid),
                with and without the LLM router on top.
                notebook: RAG/rag.ipynb -> RAG/output/{rag_only,rag_llm}/

PARAM/          Stage 3 only: parameter extraction given the correct tool (no retrieval/routing).
                notebook: PARAM/param.ipynb -> PARAM/output/

ALL/            Full pipeline: reranker -> LLM router -> LLM parameter-extraction agent.
                notebook: ALL/all.ipynb -> ALL/output/
                script:   ALL/benchmarked.py runs the same pipeline across multiple local
                          Ollama models in one pass -> ALL/output/benchmark/

input/
  documentation/tools/        One JSON schema per CLI tool (name, description, tags, parameters).
  documentation/tools_updated/ Revised/updated tool schemas.
  param/                      Query sets used by PARAM and ALL (by difficulty, language, vocab
                               level, etc.), each mapping a prompt to an expected_tool and
                               expected_params.
  tool/                       Query sets used by RAG (retrieval-only, no parameter ground truth).
```

Each notebook/script writes its results as JSON next to its own `output/` folder, so a given run's
results always live alongside the stage that produced them.

## Setup

```bash
pip install -r requirements.txt
```

Additional requirements:
- A local [Ollama](https://ollama.com) server with the models under test pulled
  (e.g. `ollama pull llama3.1:8b`).
- A CUDA-capable GPU for the `BAAI/bge-reranker-v2-m3` cross-encoder and embedding models.

## Running a benchmark

Each notebook exposes its configuration as plain variables in the first code cell — edit these,
then run all cells:

- `RAG/rag.ipynb`: `rag_mode` (`BM25` / `Vector` / `cross_encoder` / `hybrid`), `enable_llm`,
  `queries_type`, `top_k`.
- `PARAM/param.ipynb`: `model`, `queries_type`.
- `ALL/all.ipynb`: `model`, `queries_type`.

`ALL/benchmarked.py` runs the full pipeline over a list of Ollama models in one go:

```bash
python3 ALL/benchmarked.py
```

Edit `MODELS_TO_TEST` and `queries_type` at the top of the file before running.

## Results format

Every run produces a JSON summary with `metrics` (accuracy/latency aggregates) and `details`
(per-query breakdown: prompt, expected vs. selected tool/parameters). `ALL/benchmarked.py`
additionally writes a `GLOBAL_BENCHMARK_<queries_type>.json` comparing all tested models plus the
shared reranker accuracy.
