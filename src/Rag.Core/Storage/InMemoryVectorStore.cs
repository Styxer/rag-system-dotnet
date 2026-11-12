using System.Collections.Concurrent;
using System.Text.RegularExpressions;
using Rag.Core.Security;

namespace Rag.Core.Storage;

/// <summary>
/// Thread-safe, process-local vector store. Useful for local development without Postgres
/// and for tests. Keyword relevance is the fraction of query terms found in the chunk.
/// </summary>
public sealed partial class InMemoryVectorStore : IVectorStore
{
    private const int HybridCandidateMultiplier = 4;

    private readonly ConcurrentDictionary<string, VectorRecord> _records = new();
    private readonly Lock _writeLock = new();

    public int Count => _records.Count;

    public Task InitializeAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    public Task UpsertAsync(IReadOnlyCollection<VectorRecord> records, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(records);
        foreach (var record in records)
            _records[record.Id] = record;
        return Task.CompletedTask;
    }

    public Task ReplaceSourceAsync(
        string source,
        IReadOnlyCollection<VectorRecord> records,
        AccessScope scope,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(records);
        ArgumentNullException.ThrowIfNull(scope);

        lock (_writeLock)
        {
            OwnershipRules.EnsureCanReplace(scope, source, ExistingOwner(source), records);

            foreach (var (id, record) in _records)
            {
                if (record.Source == source)
                    _records.TryRemove(id, out _);
            }
            foreach (var record in records)
                _records[record.Id] = record;
        }
        return Task.CompletedTask;
    }

    public Task<int> DeleteBySourceAsync(string source, AccessScope scope, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(scope);

        lock (_writeLock)
        {
            OwnershipRules.EnsureCanDelete(scope, source, ExistingOwner(source));

            var removed = 0;
            foreach (var (id, record) in _records)
            {
                if (record.Source == source && _records.TryRemove(id, out _))
                    removed++;
            }
            return Task.FromResult(removed);
        }
    }

    public Task<IReadOnlyList<SearchResult>> SearchAsync(
        ReadOnlyMemory<float> queryEmbedding,
        int topK,
        AccessScope scope,
        CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(topK);
        ArgumentNullException.ThrowIfNull(scope);

        IReadOnlyList<SearchResult> results = Visible(scope)
            .Select(r => ToResult(r, VectorMath.CosineSimilarity(queryEmbedding.Span, r.Embedding.Span)))
            .OrderByDescending(r => r.Score)
            .ThenBy(r => r.Id, StringComparer.Ordinal)
            .Take(topK)
            .ToList();

        return Task.FromResult(results);
    }

    public Task<IReadOnlyList<SearchResult>> HybridSearchAsync(
        ReadOnlyMemory<float> queryEmbedding,
        string queryText,
        int topK,
        HybridSearchWeights weights,
        AccessScope scope,
        CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(topK);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(scope);

        var pool = topK * HybridCandidateMultiplier;
        var queryTerms = Tokenize(queryText).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var records = Visible(scope).ToList();

        var vectorRanks = records
            .Select(r => (r.Id, Score: VectorMath.CosineSimilarity(queryEmbedding.Span, r.Embedding.Span)))
            .OrderByDescending(x => x.Score)
            .ThenBy(x => x.Id, StringComparer.Ordinal)
            .Take(pool)
            .Select((x, i) => (x.Id, Rank: i + 1))
            .ToDictionary(x => x.Id, x => x.Rank);

        var keywordRanks = records
            .Select(r => (r.Id, Score: KeywordScore(queryTerms, r.Text)))
            .Where(x => x.Score > 0)
            .OrderByDescending(x => x.Score)
            .ThenBy(x => x.Id, StringComparer.Ordinal)
            .Take(pool)
            .Select((x, i) => (x.Id, Rank: i + 1))
            .ToDictionary(x => x.Id, x => x.Rank);

        IReadOnlyList<SearchResult> results = records
            .Where(r => vectorRanks.ContainsKey(r.Id) || keywordRanks.ContainsKey(r.Id))
            .Select(r => ToResult(r, weights.Fuse(
                vectorRanks.TryGetValue(r.Id, out var v) ? v : null,
                keywordRanks.TryGetValue(r.Id, out var k) ? k : null)))
            .OrderByDescending(r => r.Score)
            .ThenBy(r => r.Id, StringComparer.Ordinal)
            .Take(topK)
            .ToList();

        return Task.FromResult(results);
    }

    private IEnumerable<VectorRecord> Visible(AccessScope scope) =>
        _records.Values.Where(r => scope.CanRead(r.AllowedPrincipals));

    private string? ExistingOwner(string source) =>
        _records.Values.FirstOrDefault(r => r.Source == source)?.Owner;

    /// <summary>Fraction of distinct query terms that appear in the text (0..1).</summary>
    internal static double KeywordScore(IReadOnlySet<string> queryTerms, string text)
    {
        if (queryTerms.Count == 0)
            return 0;

        var textTerms = Tokenize(text).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var matches = queryTerms.Count(textTerms.Contains);
        return (double)matches / queryTerms.Count;
    }

    private static IEnumerable<string> Tokenize(string text) =>
        WordRegex().Matches(text ?? string.Empty)
            .Select(m => m.Value)
            .Where(w => w.Length > 2);

    private static SearchResult ToResult(VectorRecord r, double score) =>
        new(r.Id, r.Source, r.ChunkIndex, r.Text, score);

    [GeneratedRegex(@"[\p{L}\p{N}]+")]
    private static partial Regex WordRegex();
}
