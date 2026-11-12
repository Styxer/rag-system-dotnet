using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Rag.Core.Chunking;
using Rag.Core.Security;
using Rag.Core.Storage;

namespace Rag.Core.Indexing;

public sealed record IndexingResult(string DocumentId, int ChunkCount);

public interface IDocumentIndexer
{
    Task<IndexingResult> IndexDocumentAsync(
        string documentId,
        string content,
        AccessScope scope,
        IReadOnlyCollection<string>? sharedWith = null,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Indexing pipeline: chunk → embed (batched) → store.
/// Re-indexing replaces the document's old chunks in one atomic step, so a shorter new version
/// leaves no stale chunks behind and a failure leaves the old version intact.
/// </summary>
public sealed class DocumentIndexer : IDocumentIndexer
{
    internal const int EmbeddingBatchSize = 64;

    private readonly ITextChunker _chunker;
    private readonly IEmbeddingGenerator<string, Embedding<float>> _embeddingGenerator;
    private readonly IVectorStore _vectorStore;
    private readonly ILogger<DocumentIndexer> _logger;

    public DocumentIndexer(
        ITextChunker chunker,
        IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator,
        IVectorStore vectorStore,
        ILogger<DocumentIndexer>? logger = null)
    {
        _chunker = chunker;
        _embeddingGenerator = embeddingGenerator;
        _vectorStore = vectorStore;
        _logger = logger ?? NullLogger<DocumentIndexer>.Instance;
    }

    public async Task<IndexingResult> IndexDocumentAsync(
        string documentId,
        string content,
        AccessScope scope,
        IReadOnlyCollection<string>? sharedWith = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(documentId);
        ArgumentException.ThrowIfNullOrWhiteSpace(content);
        ArgumentNullException.ThrowIfNull(scope);

        var owner = scope.OwnerPrincipal;
        var acl = scope.BuildAcl(sharedWith);

        var chunks = _chunker.Chunk(documentId, content);
        var records = new List<VectorRecord>(chunks.Count);

        foreach (var batch in chunks.Chunk(EmbeddingBatchSize))
        {
            var embeddings = await _embeddingGenerator.GenerateAsync(
                batch.Select(c => c.Text),
                cancellationToken: cancellationToken);

            if (embeddings.Count != batch.Length)
                throw new InvalidOperationException(
                    $"Embedding generator returned {embeddings.Count} embeddings for {batch.Length} chunks.");

            records.AddRange(batch.Select((chunk, i) =>
                new VectorRecord(chunk.Id, chunk.SourceId, chunk.Index, chunk.Text, embeddings[i].Vector, owner, acl)));
        }

        await _vectorStore.ReplaceSourceAsync(documentId, records, scope, cancellationToken);

        _logger.LogInformation("Indexed document {DocumentId} into {ChunkCount} chunks", documentId, records.Count);
        return new IndexingResult(documentId, records.Count);
    }
}
