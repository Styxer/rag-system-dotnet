using System.Text.RegularExpressions;
using Microsoft.Extensions.AI;

namespace Rag.Testing;

/// <summary>
/// Deterministic, offline embedding generator: a hashed bag-of-words vector.
/// Texts that share words get a higher cosine similarity, which is enough to test retrieval.
/// </summary>
public sealed partial class FakeEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    public FakeEmbeddingGenerator(int dimensions = 64) => Dimensions = dimensions;

    public int Dimensions { get; }

    /// <summary>Number of GenerateAsync calls (i.e. batches).</summary>
    public int CallCount { get; private set; }

    /// <summary>Every input that was embedded, in order.</summary>
    public List<string> Inputs { get; } = [];

    /// <summary>When set, the generator returns this many embeddings regardless of the input count.</summary>
    public int? ForcedResultCount { get; set; }

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        CallCount++;
        var list = values.ToList();
        Inputs.AddRange(list);

        var embeddings = list.Select(v => new Embedding<float>(Embed(v)));
        if (ForcedResultCount is { } count)
            embeddings = embeddings.Take(count);

        return Task.FromResult(new GeneratedEmbeddings<Embedding<float>>(embeddings.ToList()));
    }

    public float[] Embed(string text)
    {
        var vector = new float[Dimensions];
        foreach (Match match in WordRegex().Matches(text.ToLowerInvariant()))
        {
            var bucket = (int)(StableHash(match.Value) % (uint)Dimensions);
            vector[bucket] += 1f;
        }
        return vector;
    }

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceKey is null && serviceType.IsInstanceOfType(this) ? this : null;

    public void Dispose() { }

    private static uint StableHash(string value)
    {
        // FNV-1a: stable across processes, unlike string.GetHashCode().
        var hash = 2166136261u;
        foreach (var c in value)
        {
            hash ^= c;
            hash *= 16777619u;
        }
        return hash;
    }

    [GeneratedRegex(@"[\p{L}\p{N}]+")]
    private static partial Regex WordRegex();
}
