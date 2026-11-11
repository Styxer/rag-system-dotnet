namespace Rag.Core;

public sealed class RagOptions
{
    public const string SectionName = "Rag";

    /// <summary>Number of chunks passed to the model as context.</summary>
    public int TopK { get; set; } = 5;

    /// <summary>When re-ranking, this many candidates are retrieved first (TopK × multiplier).</summary>
    public int RerankCandidateMultiplier { get; set; } = 4;

    /// <summary>
    /// Minimum cosine similarity for vector search; weaker chunks are dropped before re-ranking.
    /// 0 disables it. Not applied to hybrid search, whose scores are rank-based.
    /// </summary>
    public double MinimumScore { get; set; }

    public double VectorWeight { get; set; } = 0.7;
    public double KeywordWeight { get; set; } = 0.3;

    public bool UseHybridSearchByDefault { get; set; }
    public bool UseRerankingByDefault { get; set; }
}

/// <summary>Per-request overrides. Null means "use the configured default".</summary>
public sealed record QueryOptions(int? TopK = null, bool? UseHybridSearch = null, bool? UseReranking = null);

public sealed record SourceReference(string DocumentName, int ChunkIndex, double Relevance, string Excerpt);

public sealed record RagResponse(string Answer, IReadOnlyList<SourceReference> Sources);
