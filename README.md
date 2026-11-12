# RAG System in .NET

A Retrieval-Augmented Generation (RAG) API built with **ASP.NET Core 10**, **Microsoft.Extensions.AI** and **PostgreSQL + pgvector**, based on Milan Jovanović's article [Building a RAG System in .NET](https://milanjovanovic.tech/blog/rag-system-dotnet).

Upload documents, ask questions in natural language, and get answers grounded **only** in your documents, along with the sources each answer came from.

## How it works

```
Indexing:  document ──► sentence-aware chunker ──► embeddings (batched) ──► pgvector (HNSW, cosine)
Querying:  question ──► embedding ──► vector / hybrid search ──► (optional) LLM re-rank ──► grounded prompt ──► answer + sources
```

| Piece | Implementation |
|---|---|
| Chunking | Splits on `. ` `! ` `? ` and line breaks (so list items, headings and table rows stay whole), packs sentences into ~2000-char chunks (≈500 tokens) with ~200 chars of sentence overlap. Chunk ids: `{sourceId}-chunk-{index}`. Over-long sentences are hard-split. |
| Embeddings | `IEmbeddingGenerator<string, Embedding<float>>` (OpenAI `text-embedding-3-small`, 1536 dims), batched 64 chunks per call |
| Vector store | `document_embeddings` table, `vector(1536)` column, **HNSW** index with `vector_cosine_ops` (it can be built on an empty table, unlike IVFFlat), upserts with `ON CONFLICT (id) DO UPDATE` |
| Retrieval | Cosine similarity (`1 - (embedding <=> query)`), top-K = 5 |
| Hybrid search | Vector ranking and `ts_rank` full-text ranking merged with a `FULL OUTER JOIN` and fused by weighted **Reciprocal Rank Fusion** (0.7 / 0.3). Ranks are combined, not raw scores, because cosine similarity and `ts_rank` are on very different scales |
| Re-ranking | The LLM scores each candidate's first 200 chars from 0.0 to 1.0 (JSON array). If the reply can't be parsed, the original order is kept |
| Generation | `IChatClient` (OpenAI `gpt-4o`). The system prompt allows answers **only from the context**, otherwise *"I don't have information about that in my knowledge base."* Retrieved chunks go in the user message inside `<context>` tags and are treated as data, so text in an uploaded document cannot act as an instruction |

Beyond the article:
- `POST /api/query` returns **source references** (document, chunk index, relevance, excerpt).
- Re-indexing a document **replaces its chunks in one transaction**: no stale chunks, and a failed re-index keeps the old version.
- Searches raise `hnsw.ef_search` to the number of candidates requested, so large `topK` / re-rank pools are not cut at pgvector's default of 40.
- Startup fails with a clear message if the OpenAI key is missing or the table's vector size does not match `PgVector:Dimensions`.
- If retrieval finds nothing, the API answers "no information" **without calling the LLM**.
- Optional `MinimumScore` similarity threshold, per-request `topK` / hybrid / re-rank switches.
- Swappable `IVectorStore`: `Postgres` (default) or `InMemory` (no database needed).
- Works with any **OpenAI-compatible endpoint** (`OpenAI:Endpoint`), e.g. Azure OpenAI or a local model server.

## Project layout

```
src/
  Rag.Core            Chunking, indexing, retrieval, re-ranking, prompt building, RagService (no infrastructure deps)
  Rag.Infrastructure  PgVectorStore (Npgsql + Pgvector)
  Rag.Api             Minimal API, DI wiring, OpenAI clients
tests/
  Rag.Testing         Offline fakes: deterministic bag-of-words embedding generator, scripted chat client
  Rag.Core.Tests      Unit tests (chunker, vector math, in-memory store, indexer, re-ranker, prompt, RAG pipeline)
  Rag.Api.Tests       Endpoint tests with WebApplicationFactory (no DB / no API key needed)
  Rag.IntegrationTests  PgVectorStore against a real pgvector container (Testcontainers, needs Docker)
```

## Getting started

Prerequisites: .NET 10 SDK, Docker, an OpenAI API key.

```bash
# 1. Start Postgres with pgvector
docker compose up -d

# 2. Set your API key (kept out of source control)
dotnet user-secrets set "OpenAI:ApiKey" "sk-..." --project src/Rag.Api

# 3. Run the API (schema is created on startup)
dotnet run --project src/Rag.Api
```

The API listens on `http://localhost:5080`. Try the requests in [`src/Rag.Api/Rag.Api.http`](src/Rag.Api/Rag.Api.http), or:

```bash
curl -X POST http://localhost:5080/api/documents \
  -H "Content-Type: application/json" \
  -d '{"fileName":"handbook.md","content":"Employees receive twenty five vacation days per year."}'

curl -X POST http://localhost:5080/api/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How many vacation days do I get?","useHybridSearch":true}'
```

Response:

```json
{
  "answer": "Employees receive twenty five vacation days per year.",
  "sources": [
    { "documentName": "handbook.md", "chunkIndex": 0, "relevance": 0.83, "excerpt": "Employees receive twenty five vacation days per year." }
  ]
}
```

No database at hand? Run with the in-memory store (the OpenAI key is still needed):

```bash
dotnet run --project src/Rag.Api -- --VectorStore:Provider=InMemory
```

## API

All routes except `/health` need a signed-in user (`Authorization: Bearer <jwt>`).

| Method | Route | Body | Returns |
|---|---|---|---|
| `POST` | `/api/documents` | `{ fileName, content, allowedPrincipals? }` | `{ message, documentId, chunkCount }` |
| `POST` | `/api/query` | `{ question, topK?, useHybridSearch?, useReranking? }` | `{ answer, sources[] }` |
| `GET` | `/health` | none (anonymous) | `{ status }` |

Status codes: `400` invalid input (`ProblemDetails`), `401` not signed in, `403` token has no `sub` claim, `409` the file name belongs to another owner, `429` rate limit hit (with `Retry-After`). Documents are limited to 1,000,000 characters and file names to 200.

## Security

**Sign-in.** The API validates JWTs from your identity provider. Set `Auth:Authority` (issuer URL) and `Auth:Audience`; outside Development the app refuses to start without an authority. The user id is the `sub` claim and groups come from the claim named in `Auth:GroupClaim` (default `groups`). For local work, create a token with the .NET tooling:

```bash
dotnet user-jwts create --project src/Rag.Api --claim groups=hr
```

**Per-document access control.** Every chunk stores an owner and an access list of principals (`user:<id>`, `group:<name>`). The list is applied *inside* the SQL search query, so text the caller may not read never reaches the prompt or the response. Every store and service method takes an `AccessScope` explicitly; there is no way to search without one.

- A document is private to its uploader unless `allowedPrincipals` lists others (for example `["group:hr", "user:bob"]`). The uploader always keeps access.
- The file name is the document id and belongs to whoever uploaded it first. Uploading the same name as someone else returns `409`; the owner can re-upload to replace the text or change the sharing.
- Rows created before access control existed get the system owner and an empty list, so they are invisible to every user until re-indexed.
- Filtering uses a GIN index on the access list. With pgvector 0.8+ the store enables iterative HNSW scans, so a filtered search still returns up to `topK` rows; older pgvector versions may return fewer when most nearby chunks are forbidden.
- There is no endpoint yet to delete a document or change its sharing without re-uploading the content.

**Rate limits.** Per user, per minute: `RateLimiting:QueryPerMinute` (30) and `RateLimiting:UploadPerMinute` (10). The counters live in memory, so with several instances each one counts separately; use a shared limiter (for example at the gateway) when you scale out.

**Secrets.** The OpenAI key and connection string come from user-secrets or environment variables, never from the repository. A secret store such as Key Vault is deliberately not wired in: it adds a dependency and a cloud account this project does not require, and environment variables fed by your platform's secret store cover the same need.

## Configuration (`appsettings.json`)

| Key | Default | Notes |
|---|---|---|
| `ConnectionStrings:Postgres` | localhost / rag | |
| `OpenAI:ApiKey` | (none) | Use user-secrets or the `OpenAI__ApiKey` env var |
| `OpenAI:Endpoint` | (none) | Optional OpenAI-compatible base URL |
| `OpenAI:ChatModel` / `EmbeddingModel` | `gpt-4o` / `text-embedding-3-small` | |
| `VectorStore:Provider` | `Postgres` | or `InMemory` |
| `PgVector:Dimensions` | `1536` | Must match the embedding model |
| `Chunking:ChunkSize` / `ChunkOverlap` | `2000` / `200` | characters |
| `Rag:TopK` | `5` | |
| `Rag:VectorWeight` / `KeywordWeight` | `0.7` / `0.3` | hybrid (RRF) weights; tune for your data |
| `Rag:RerankCandidateMultiplier` | `4` | re-ranking retrieves TopK × this many candidates |
| `Auth:Authority` / `Auth:Audience` | (none) | Identity provider for JWT sign-in; Authority is required outside Development |
| `Auth:GroupClaim` | `groups` | Token claim listing the caller's groups |
| `Auth:RequireHttpsMetadata` | `true` | |
| `RateLimiting:QueryPerMinute` / `UploadPerMinute` | `30` / `10` | Per user |
| `Rag:MinimumScore` | `0` | Minimum cosine similarity, applied to vector search before re-ranking. 0 = off. Not applied to hybrid search (rank-based scores) |

Package versions are pinned in one place: `Directory.Packages.props`.

## Tests

```bash
# Unit + API tests (fast, offline, no Docker)
dotnet test --filter "Category!=Integration"

# Everything, including pgvector integration tests (Docker must be running)
dotnet test
```

The tests use a deterministic hashed bag-of-words `FakeEmbeddingGenerator` and a scripted `FakeChatClient`, so the whole pipeline (chunk → embed → store → retrieve → re-rank → prompt → answer) runs without network access or API keys.

## When RAG is the wrong tool

As the article points out: if the whole knowledge base fits in the prompt, just put it there. For questions over structured data, use function calling or SQL. For whole-corpus aggregation ("summarize every document"), retrieval of top-K chunks won't cover it.
