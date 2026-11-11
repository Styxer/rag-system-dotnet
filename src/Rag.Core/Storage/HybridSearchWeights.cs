namespace Rag.Core.Storage;

/// <summary>
/// Weights for hybrid search. Results are fused with weighted Reciprocal Rank Fusion (RRF):
/// each list contributes <c>weight / (RrfK + rank)</c>, so only the position in each list matters,
/// not the raw scores (cosine similarity and full-text rank live on very different scales).
/// The fused score is scaled so that a chunk ranked first in both lists scores 1.0.
/// </summary>
public sealed record HybridSearchWeights(double VectorWeight = 0.7, double KeywordWeight = 0.3)
{
    /// <summary>Standard RRF constant; dampens the gap between neighbouring ranks.</summary>
    public const int RrfK = 60;

    public static HybridSearchWeights Default { get; } = new();

    /// <summary>Multiplier that makes "rank 1 in both lists" equal 1.0.</summary>
    public double Scale
    {
        get
        {
            if (VectorWeight < 0 || KeywordWeight < 0 || VectorWeight + KeywordWeight <= 0)
                throw new InvalidOperationException("Hybrid search weights must be non-negative and not both zero.");
            return (RrfK + 1) / (VectorWeight + KeywordWeight);
        }
    }

    /// <summary>Fuses 1-based ranks. Null means the chunk is missing from that list.</summary>
    public double Fuse(int? vectorRank, int? keywordRank)
    {
        var vector = vectorRank is { } v ? VectorWeight / (RrfK + v) : 0;
        var keyword = keywordRank is { } k ? KeywordWeight / (RrfK + k) : 0;
        return (vector + keyword) * Scale;
    }
}
