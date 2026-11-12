using Rag.Core.Security;

namespace Rag.Core.Storage;

/// <summary>
/// A chunk together with its embedding, ready to be stored.
/// <paramref name="Owner"/> is the principal that uploaded the document and
/// <paramref name="AllowedPrincipals"/> is the access list; a chunk with an empty list is
/// visible only to <see cref="AccessScope.Unrestricted"/>.
/// </summary>
public sealed record VectorRecord(
    string Id,
    string Source,
    int ChunkIndex,
    string Text,
    ReadOnlyMemory<float> Embedding,
    string Owner = AccessScope.SystemOwnerPrincipal,
    IReadOnlyList<string>? AllowedPrincipals = null);

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

    /// <summary>
    /// Inserts the records, replacing existing records with the same id. This does no ownership check:
    /// it is for trusted internal use (seeding, tests). Request handling must use <see cref="ReplaceSourceAsync"/>.
    /// </summary>
    Task UpsertAsync(IReadOnlyCollection<VectorRecord> records, CancellationToken cancellationToken = default);

    /// <summary>
    /// Replaces every chunk of a source document with <paramref name="records"/> as one atomic step,
    /// so a failed re-index never leaves the document missing or half-written.
    /// Throws <see cref="SourceOwnershipException"/> when the document belongs to someone else.
    /// </summary>
    Task ReplaceSourceAsync(
        string source,
        IReadOnlyCollection<VectorRecord> records,
        AccessScope scope,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Removes every chunk of a source document and returns how many were removed.
    /// Throws <see cref="SourceOwnershipException"/> when the document belongs to someone else.
    /// </summary>
    Task<int> DeleteBySourceAsync(string source, AccessScope scope, CancellationToken cancellationToken = default);

    /// <summary>
    /// Pure semantic search: cosine similarity between the query embedding and stored embeddings,
    /// over the chunks the caller may read.
    /// </summary>
    Task<IReadOnlyList<SearchResult>> SearchAsync(
        ReadOnlyMemory<float> queryEmbedding,
        int topK,
        AccessScope scope,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Hybrid search: fuses the vector ranking and the keyword (full-text) ranking with
    /// weighted Reciprocal Rank Fusion. Scores are rank-based, in (0, 1].
    /// Only chunks the caller may read are considered.
    /// </summary>
    Task<IReadOnlyList<SearchResult>> HybridSearchAsync(
        ReadOnlyMemory<float> queryEmbedding,
        string queryText,
        int topK,
        HybridSearchWeights weights,
        AccessScope scope,
        CancellationToken cancellationToken = default);
}
