using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Rag.Core.Generation;
using Rag.Core.Retrieval;
using Rag.Core.Security;
using Rag.Core.Storage;

namespace Rag.Core;

public interface IRagService
{
    Task<RagResponse> AskAsync(string question, AccessScope scope, QueryOptions? options = null, CancellationToken cancellationToken = default);
}

/// <summary>
/// Query pipeline: embed the question → retrieve (vector or hybrid) → optional LLM re-rank
/// → build a grounded prompt → generate the answer, returning the sources that were used.
/// </summary>
public sealed class RagService : IRagService
{
    internal const int ExcerptLength = 200;

    private readonly IEmbeddingGenerator<string, Embedding<float>> _embeddingGenerator;
    private readonly IVectorStore _vectorStore;
    private readonly IChatClient _chatClient;
    private readonly IReranker _reranker;
    private readonly RagOptions _options;
    private readonly ILogger<RagService> _logger;

    public RagService(
        IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator,
        IVectorStore vectorStore,
        IChatClient chatClient,
        IReranker reranker,
        IOptions<RagOptions> options,
        ILogger<RagService>? logger = null)
    {
        _embeddingGenerator = embeddingGenerator;
        _vectorStore = vectorStore;
        _chatClient = chatClient;
        _reranker = reranker;
        _options = options.Value;
        _logger = logger ?? NullLogger<RagService>.Instance;
    }

    public async Task<RagResponse> AskAsync(
        string question,
        AccessScope scope,
        QueryOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(question);
        ArgumentNullException.ThrowIfNull(scope);

        var topK = options?.TopK ?? _options.TopK;
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(topK);

        var useHybrid = options?.UseHybridSearch ?? _options.UseHybridSearchByDefault;
        var useReranking = options?.UseReranking ?? _options.UseRerankingByDefault;
        var candidateCount = useReranking ? topK * Math.Max(1, _options.RerankCandidateMultiplier) : topK;

        var queryEmbedding = await _embeddingGenerator.GenerateVectorAsync(question, cancellationToken: cancellationToken);

        var results = useHybrid
            ? await _vectorStore.HybridSearchAsync(
                queryEmbedding,
                question,
                candidateCount,
                new HybridSearchWeights(_options.VectorWeight, _options.KeywordWeight),
                scope,
                cancellationToken)
            : await _vectorStore.SearchAsync(queryEmbedding, candidateCount, scope, cancellationToken);

        // The threshold is a cosine similarity, so it only makes sense on plain vector results.
        if (!useHybrid && _options.MinimumScore > 0)
            results = results.Where(r => r.Score >= _options.MinimumScore).ToList();

        if (useReranking && results.Count > 0)
            results = await _reranker.RerankAsync(question, results, topK, cancellationToken);

        _logger.LogInformation(
            "Retrieved {Count} chunks (hybrid: {Hybrid}, rerank: {Rerank})", results.Count, useHybrid, useReranking);

        if (results.Count == 0)
            return new RagResponse(PromptBuilder.NoInformationAnswer, []);

        var messages = PromptBuilder.Build(question, results);
        var response = await _chatClient.GetResponseAsync(messages, cancellationToken: cancellationToken);

        var sources = results
            .Select(r => new SourceReference(r.Source, r.ChunkIndex, Math.Round(r.Score, 4), Excerpt(r.Text)))
            .ToList();

        return new RagResponse(response.Text, sources);
    }

    private static string Excerpt(string text) =>
        text.Length <= ExcerptLength ? text : text[..ExcerptLength] + "…";
}
