namespace Rag.Core.Storage;

/// <summary>A chunk together with its embedding, ready to be stored.</summary>
public sealed record VectorRecord(
    string Id,
    string Source,
    int ChunkIndex,
    string Text,
    ReadOnlyMemory<float> Embedding);

/// <summary>A stored chunk returned by a search, with a relevance score (higher is better).</summary>
public sealed record SearchResult(
    string Id,
    string Source,
    int ChunkIndex,
    string Text,
    double Score);

public interface IVectorStore
{
    /// <summary>Creates tables/indexes if they do not exist yet.</summary>
    Task InitializeAsync(CancellationToken cancellationToken = default);

    /// <summary>Inserts the records, replacing existing records with the same id.</summary>
    Task UpsertAsync(IReadOnlyCollection<VectorRecord> records, CancellationToken cancellationToken = default);

    /// <summary>
    /// Replaces every chunk of a source document with <paramref name="records"/> as one atomic step,
    /// so a failed re-index never leaves the document missing or half-written.
    /// </summary>
    Task ReplaceSourceAsync(string source, IReadOnlyCollection<VectorRecord> records, CancellationToken cancellationToken = default);

    /// <summary>Removes every chunk of a source document. Returns the number of removed chunks.</summary>
    Task<int> DeleteBySourceAsync(string source, CancellationToken cancellationToken = default);

    /// <summary>Pure semantic search: cosine similarity between the query embedding and stored embeddings.</summary>
    Task<IReadOnlyList<SearchResult>> SearchAsync(
        ReadOnlyMemory<float> queryEmbedding,
        int topK,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Hybrid search: fuses the vector ranking and the keyword (full-text) ranking with
    /// weighted Reciprocal Rank Fusion. Scores are rank-based, in (0, 1].
    /// </summary>
    Task<IReadOnlyList<SearchResult>> HybridSearchAsync(
        ReadOnlyMemory<float> queryEmbedding,
        string queryText,
        int topK,
        HybridSearchWeights weights,
        CancellationToken cancellationToken = default);
}
