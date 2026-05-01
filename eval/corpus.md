# Eval Corpus

_Generated from `dataset.json` contexts. Each section is the ground-truth context for one question in the golden set._


## Topic 1: What is LangGraph and what problem does it solve?

LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. It provides a way to define flows that involve cycles, which is essential for most agentic architectures.


## Topic 2: How does LangGraph handle state management?

In LangGraph, state is defined as a TypedDict and shared across all nodes in the graph. Each node function receives the current state as input and returns a dictionary of updates. Reducers (like operator.add for lists) define how returned values are combined with existing state.


## Topic 3: What is the difference between edges and conditional edges in LangGraph?

Edges connect nodes in the graph. A normal edge always goes to the same next node. A conditional edge calls a function that examines the state and returns the name of the next node to visit. This is how you implement decision points like should_continue functions.


## Topic 4: How do you add tool calling to a LangGraph agent?

Tools are added by calling model.bind_tools(tools) and creating a ToolNode. The graph routes between the agent node (which calls the LLM) and the tool node using a conditional edge. The routing function checks if the last message has tool_calls.


## Topic 5: What is a checkpoint in LangGraph?

Checkpointing in LangGraph saves a snapshot of the graph state after each node execution. The MemorySaver stores checkpoints in memory. SqliteSaver persists them to a database. Checkpoints enable features like rewinding to previous states and resuming interrupted workflows.


## Topic 6: How does RAG retrieval work with vector databases?

RAG retrieval involves embedding documents and queries into the same vector space. The vector database performs approximate nearest neighbor search to find chunks whose embeddings are closest to the query embedding. Common similarity metrics include cosine similarity and dot product.


## Topic 7: What is hybrid retrieval and why is it better than dense-only search?

Dense retrieval uses vector embeddings to capture semantic meaning but can miss exact keyword matches. BM25 is a statistical keyword-based method that excels at matching specific terms. Hybrid retrieval combines both using fusion methods like Reciprocal Rank Fusion (RRF), where the final score is computed as 1/(k+rank) summed across both result lists.


## Topic 8: What metrics does RAGAS use to evaluate RAG systems?

RAGAS evaluates RAG pipelines with metrics including faithfulness, which measures whether the generated answer is supported by the retrieved context; answer relevancy, which measures how well the answer addresses the question; context precision, which measures the proportion of retrieved chunks that are relevant; and context recall, which measures whether all ground-truth relevant information was retrieved.


## Topic 9: What is chunking and how does chunk size affect retrieval quality?

Document chunking divides text into segments that can be independently embedded and retrieved. Chunk size affects the granularity of retrieval. Small chunks (128-256 tokens) enable precise matching but may miss surrounding context. Large chunks (512-1024 tokens) provide more context but can dilute relevance. Overlapping chunks help preserve context at boundaries.


## Topic 10: How does Reciprocal Rank Fusion work?

Reciprocal Rank Fusion (RRF) is a method for combining multiple ranked lists. For each document, the RRF score is calculated as the sum of 1/(k+rank_i) across all lists where it appears. The constant k (default 60) controls how much top-ranked items are favored. Documents are then sorted by their combined RRF score.


## Topic 11: What is faithfulness in RAGAS and how is it measured?

RAGAS faithfulness is computed by first decomposing the generated answer into individual factual claims, then using an LLM judge to verify whether each claim can be inferred from the retrieved context. The metric returns the proportion of supported claims, on a scale from 0 to 1.


## Topic 12: How does answer relevancy differ from faithfulness?

Faithfulness asks: is the answer grounded in the retrieved evidence? Answer relevancy asks: does the answer actually address the user's question? RAGAS computes answer relevancy by generating synthetic questions from the answer and measuring their similarity to the original question.


## Topic 13: What is context precision and why does it matter?

Context precision in RAGAS evaluates how relevant each retrieved chunk is to the input question, weighted by rank. The metric ranges from 0 to 1; high precision means the top-ranked retrieved contexts are truly useful for answering the question.


## Topic 14: What is context recall and how is it computed?

Context recall is calculated by decomposing the reference answer into individual statements and checking, for each one, whether supporting context was present in the retrieved chunks. It directly measures retrieval coverage and complements context precision.


## Topic 15: Why use BM25 alongside dense embeddings instead of relying on dense alone?

Dense retrieval may fail on out-of-vocabulary terms, exact identifiers, or domain-specific tokens that the embedding model has not seen at scale. BM25 is a token-overlap based method that performs reliably on such queries. Hybrid setups exploit both signals for robust recall.


## Topic 16: How does Qdrant differ from FAISS or Chroma?

Qdrant offers a Rust-based vector database with HNSW indexing, strict filtering, snapshots, and clustering support. FAISS is Meta's library focused on raw search speed without server features. Chroma provides an embedded SQLite-backed vector store optimized for local prototyping.


## Topic 17: What is HNSW and why is it used in vector databases?

HNSW indexes vectors by constructing layered proximity graphs. Higher layers have sparser connections for coarse navigation, lower layers are denser for fine-grained search. Most production vector databases including Qdrant and Weaviate use HNSW as their default index.


## Topic 18: What does cosine similarity measure between two embeddings?

Cosine similarity computes the dot product of two vectors divided by the product of their L2 norms. It is direction-invariant to magnitude, which matters for embeddings since absolute vector length carries no semantic meaning.


## Topic 19: Why are embeddings often normalized to unit length before storage?

Unit-length normalization (L2 normalization) divides each embedding by its norm. With normalized vectors, cosine similarity equals the dot product, allowing the database to use the cheaper dot-product kernel. Many embedding APIs (OpenAI, sentence-transformers) return pre-normalized vectors.


## Topic 20: What is sentence-transformers all-MiniLM-L6-v2 and when should you use it?

all-MiniLM-L6-v2 is a 22M-parameter MiniLM checkpoint fine-tuned on sentence-pair tasks. It produces 384-dimensional embeddings suitable for semantic search and similarity ranking. Its small footprint makes it the standard low-cost embedding model for self-hosted RAG.


## Topic 21: What does it mean to embed a query and a document into the same vector space?

Embedding models are trained so that semantically related sentences map to nearby vectors. By using the same model for both indexing and querying, the system ensures vectors are directly comparable. This is the foundation of dense retrieval.


## Topic 22: What is query rewriting and why does it improve RAG?

Query rewriting uses an LLM to transform the original user question, often by adding clarifying terms, resolving pronouns from chat history, or generating multiple paraphrases. The rewritten queries drive retrieval and frequently boost recall on conversational or under-specified inputs.


## Topic 23: Why does pre-computing embeddings asynchronously help latency?

Asynchronous embedding pre-computation decouples document arrival from vector indexing. A background worker pulls newly uploaded chunks from a queue and writes their embeddings to the vector database without blocking the ingest API. This pattern keeps p95 ingest latency low even on large uploads.


## Topic 24: What is semantic caching and how does it differ from a regular query cache?

Semantic cache implementations store embeddings of past queries alongside their cached responses. On a new query, the system embeds it and searches the cache for a vector whose similarity exceeds a threshold (e.g. 0.95). Hits return the cached answer; misses fall through to the full pipeline.


## Topic 25: How do you measure p95 latency for a RAG query?

Latency percentiles are computed by sorting recorded query durations and reading off the value at a given percentile rank. p50 is the median, p95 the value below which 95% of queries fall, p99 the tail-latency indicator. Production RAG systems track p95 because it represents perceived responsiveness for most users.


## Topic 26: What is MLflow used for in an LLM project?

MLflow provides experiment tracking, model registry, and serving capabilities. In LLM projects it is commonly used to record evaluation metrics (faithfulness, relevance) alongside the prompt template, retrieval mode, chunk size, and model name used for each run, enabling regression detection across iterations.


## Topic 27: How would you detect a regression in RAG quality between two MLflow runs?

Regression detection compares evaluation metrics across runs. A simple rule: for each tracked metric, compute the percentage drop versus baseline; if any drop exceeds the threshold, mark a regression. CI pipelines can be configured to fail on regression so prompt changes don't silently degrade quality.


## Topic 28: Why instrument LLM calls with tracing?

LLM tracing tools like Langfuse or LangSmith log every call's prompt, response, retrieved context, latency, and cost. Traces are linked by session and trace IDs, making it possible to reconstruct conversations and debug specific failure modes. Tracing also feeds dashboards on cost and quality drift.


## Topic 29: What's the role of structured outputs in agent reliability?

Structured outputs constrain LLM responses to a typed schema, validated either by the API (function calling) or by a Pydantic parser. This makes agent loops robust to free-form drift and ensures every step produces fields that the next step can consume without parsing exceptions.


## Topic 30: What is the difference between fine-tuning and RAG for adding knowledge to an LLM?

Fine-tuning updates a model's parameters using supervised examples, making it suitable for adapting style or learning task-specific behavior. RAG injects knowledge through retrieved context at inference time, keeping the model frozen and the knowledge base mutable. Most production document-QA systems prefer RAG for freshness and auditability.


## Topic 31: When does fine-tuning make more sense than RAG?

Use fine-tuning for style adaptation, learned classifiers, structured-output adherence, or compressing many in-context examples into the model itself. Reserve RAG for factual recall over a large or changing corpus where freshness and citation matter.


## Topic 32: What is QLoRA and why is it efficient?

QLoRA loads a pretrained model in 4-bit (NF4 quantization) and trains low-rank LoRA adapters on top. Memory usage drops by an order of magnitude compared to full fine-tuning, allowing 7B+ models to fine-tune on a single consumer GPU. Quality is typically within 1-2% of full-precision fine-tuning.


## Topic 33: What does PEFT stand for and what does it enable?

Parameter-Efficient Fine-Tuning includes LoRA, QLoRA, prefix tuning, and prompt tuning. These methods train a small number of additional parameters while keeping the base model frozen. The resulting adapters are typically a few megabytes, allowing one base model to serve many specialized variants.


## Topic 34: Why use Pydantic for LLM output validation?

Pydantic models provide runtime type checking and validation. When applied to LLM outputs (parsed from JSON), they catch missing fields, wrong types, and constraint violations. Combined with function calling or structured-output APIs, they form the backbone of reliable agent and pipeline code.


## Topic 35: What is the role of the Planner agent in a multi-agent pipeline?

Planner agents perform task decomposition, breaking an open-ended goal into a structured plan that the rest of the pipeline can execute. They emit typed sub-tasks (sub-questions, action steps) that the Researcher, Writer, or other agents consume. Planner output schema directly bounds the pipeline's reasoning shape.


## Topic 36: What does the Researcher agent typically do in a research pipeline?

The Researcher agent receives sub-questions from the Planner, queries multiple knowledge sources (web search, vector database, internal APIs), and returns evidence with citations. In parallel pipelines it runs sub-questions concurrently to minimize wall-clock time.


## Topic 37: What does the Writer agent do at the end of a pipeline?

The Writer agent consumes structured research findings and produces the final user-facing report. It is responsible for tone, structure, and citation consistency. Bounded prompts and Pydantic-typed inputs prevent it from hallucinating beyond the evidence the Researcher gathered.


## Topic 38: Why are role boundaries between agents important?

Multi-agent systems are most reliable when each agent has a narrow, well-defined role enforced by its input/output schema. Cross-role drift (planner answering, writer planning) typically produces lower-quality outputs than upgrading the model itself would fix.


## Topic 39: What is a Pydantic structured output and why use it for agent steps?

Structured outputs in agent pipelines are typed schemas (Pydantic models, JSON schemas) that each agent must produce. They serve as contracts between agents, ensuring the next stage receives validated, predictable input. This is foundational for reliable multi-agent orchestration.


## Topic 40: How does FastAPI help with serving an LLM application?

FastAPI is a Python web framework built on Starlette and Pydantic. It supports async/await natively, making it well-suited for LLM-backed applications where requests spend most of their time waiting on remote API calls. Type-driven validation and OpenAPI generation reduce boilerplate.


## Topic 41: Why use Server-Sent Events (SSE) instead of polling for streaming LLM output?

Server-Sent Events stream text/event-stream responses from server to client. They use plain HTTP, work through proxies and CDNs, and require no client-side library. For LLM streaming and agent-stage progress UI, SSE matches the unidirectional nature of the stream while avoiding WebSocket complexity.


## Topic 42: What is the difference between p50, p95, and p99 latency?

Latency percentiles describe the distribution of response times. P50 (median) reflects typical experience; p95 captures the slow 5%; p99 captures the slowest 1%. Optimizing for p95 means optimizing the experience most users would notice as slow, not just the median path.


## Topic 43: What is cold-start latency in a RAG service and how can you reduce it?

Cold-start latency in RAG comes from loading embedding/generation models, building in-memory indices like BM25, and warming caches. Strategies include rebuilding BM25 on startup from the persisted vector database, preloading the embedding model, and keeping a warm pool of process workers.


## Topic 44: Why is a singleton retriever often used in a FastAPI RAG app?

Singleton retriever patterns store one shared instance per process. The BM25 index, which can take seconds to build over thousands of chunks, is constructed once on first access. A write lock around add_documents keeps concurrent ingestion from corrupting the index.


## Topic 45: How does Docker help in deploying a RAG service?

Containers package an application's environment along with its code, ensuring consistent behavior across machines. For RAG services, Docker images typically bundle Python dependencies, model weights or fetch hooks, and app entrypoints. Container orchestrators handle deployment, health checks, and scaling.


## Topic 46: What does GitHub Actions CI do for a RAG project?

GitHub Actions provides CI/CD pipelines defined as YAML workflows. RAG project workflows commonly run pytest, optional ragas-eval on a small dataset, container builds on push to main, and deploy steps gated on green CI. This automates regression detection and release.


## Topic 47: Why include a /health endpoint in a RAG service?

A /health endpoint returns a lightweight response indicating the process is up. Readiness variants additionally verify dependencies (vector database reachable, embedder loaded) before reporting healthy. Kubernetes uses these for liveness/readiness probes; uptime monitors use them for alerting.


## Topic 48: What is prompt injection and how do you defend against it in RAG?

Prompt injection attacks embed adversarial instructions inside retrieved documents or user input, attempting to override the system prompt. Mitigations include explicit data-vs-instruction delimiters, structured-output schemas, content filtering, and never letting untrusted text dictate which tools to call.


## Topic 49: Why use a per-tenant namespace in a vector database for multi-client RAG?

Multi-tenant RAG must isolate data at the storage layer. Vector databases support this through per-tenant collections, namespaces, or payload filters scoped to a tenant identifier. Combined with authenticated request scoping, this keeps each client's corpus retrievable only by that client's queries.


## Topic 50: What is a golden test set in RAG evaluation?

Golden test sets are hand-curated evaluation datasets used as fixed benchmarks. Each entry typically contains a question, a reference answer, and optionally the supporting context. The same set is run repeatedly across changes to detect regressions and validate improvements.
