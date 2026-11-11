using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;
using Rag.Core.Chunking;
using Rag.Core.Generation;
using Rag.Core.Indexing;
using Rag.Core.Retrieval;
using Rag.Core.Storage;
using Rag.Testing;

namespace Rag.Core.Tests;

public class RagServiceTests
{
    private readonly FakeEmbeddingGenerator _embeddings = new();
    private readonly SpyVectorStore _store = new();
    private readonly FakeChatClient _chat = new("Employees receive twenty five vacation days per year.");
    private readonly SpyReranker _reranker = new();

    private RagService CreateService(RagOptions? options = null) =>
        new(_embeddings, _store, _chat, _reranker, Options.Create(options ?? new RagOptions()));

    private async Task IndexSamplesAsync()
    {
        var indexer = new DocumentIndexer(
            new TextChunker(new ChunkingOptions { ChunkSize = 120, ChunkOverlap = 0 }), _embeddings, _store);
        await indexer.IndexDocumentAsync(SampleDocuments.HandbookName, SampleDocuments.Handbook);
        await indexer.IndexDocumentAsync(SampleDocuments.RecipesName, SampleDocuments.Recipes);
    }

    [Fact]
    public async Task Ask_ReturnsModelAnswerAndRelevantSources()
    {
        await IndexSamplesAsync();

        var response = await CreateService().AskAsync("How many vacation days do employees receive per year?");

        Assert.Equal("Employees receive twenty five vacation days per year.", response.Answer);
        Assert.NotEmpty(response.Sources);
        Assert.Equal(SampleDocuments.HandbookName, response.Sources[0].DocumentName);
        Assert.Contains("vacation days", response.Sources[0].Excerpt);
    }

    [Fact]
    public async Task Ask_SendsRetrievedContextAndQuestionToModel()
    {
        await IndexSamplesAsync();

        await CreateService().AskAsync("How many vacation days do employees receive per year?");

        var request = _chat.LastRequest;
        Assert.Equal(ChatRole.System, request[0].Role);
        Assert.DoesNotContain("twenty five vacation days", request[0].Text);
        Assert.Equal(ChatRole.User, request[^1].Role);
        Assert.Contains("twenty five vacation days", request[^1].Text);
        Assert.EndsWith("How many vacation days do employees receive per year?", request[^1].Text);
    }

    [Fact]
    public async Task Ask_EmptyKnowledgeBase_ReturnsNoInformationWithoutCallingModel()
    {
        var response = await CreateService().AskAsync("Anything?");

        Assert.Equal(PromptBuilder.NoInformationAnswer, response.Answer);
        Assert.Empty(response.Sources);
        Assert.Equal(0, _chat.CallCount);
    }

    [Fact]
    public async Task Ask_UsesConfiguredTopK()
    {
        await IndexSamplesAsync();

        var response = await CreateService(new RagOptions { TopK = 2 }).AskAsync("vacation");

        Assert.Equal(2, response.Sources.Count);
        Assert.Equal(2, _store.LastTopK);
    }

    [Fact]
    public async Task Ask_RequestTopKOverridesConfiguration()
    {
        await IndexSamplesAsync();

        var response = await CreateService(new RagOptions { TopK = 2 }).AskAsync("vacation", new QueryOptions(TopK: 1));

        Assert.Single(response.Sources);
    }

    [Fact]
    public async Task Ask_DefaultUsesVectorSearch()
    {
        await IndexSamplesAsync();

        await CreateService().AskAsync("vacation");

        Assert.Equal(1, _store.SearchCalls);
        Assert.Equal(0, _store.HybridCalls);
    }

    [Fact]
    public async Task Ask_HybridRequested_UsesHybridSearchWithConfiguredWeights()
    {
        await IndexSamplesAsync();

        await CreateService(new RagOptions { VectorWeight = 0.6, KeywordWeight = 0.4 })
            .AskAsync("vacation", new QueryOptions(UseHybridSearch: true));

        Assert.Equal(0, _store.SearchCalls);
        Assert.Equal(1, _store.HybridCalls);
        Assert.Equal(new HybridSearchWeights(0.6, 0.4), _store.LastWeights);
    }

    [Fact]
    public async Task Ask_HybridEnabledByDefault_UsesHybridSearch()
    {
        await IndexSamplesAsync();

        await CreateService(new RagOptions { UseHybridSearchByDefault = true }).AskAsync("vacation");

        Assert.Equal(1, _store.HybridCalls);
    }

    [Fact]
    public async Task Ask_RerankingRequested_RetrievesWiderPoolThenReranks()
    {
        await IndexSamplesAsync();

        var response = await CreateService(new RagOptions { TopK = 2, RerankCandidateMultiplier = 3 })
            .AskAsync("vacation", new QueryOptions(UseReranking: true));

        Assert.Equal(6, _store.LastTopK);
        Assert.Equal(1, _reranker.CallCount);
        Assert.Equal(2, _reranker.LastTopK);
        Assert.Equal(2, response.Sources.Count);
    }

    [Fact]
    public async Task Ask_WithoutReranking_DoesNotCallReranker()
    {
        await IndexSamplesAsync();

        await CreateService().AskAsync("vacation");

        Assert.Equal(0, _reranker.CallCount);
    }

    [Fact]
    public async Task Ask_MinimumScore_FiltersWeakMatches()
    {
        await IndexSamplesAsync();

        var response = await CreateService(new RagOptions { MinimumScore = 0.99 })
            .AskAsync("completely unrelated quantum chromodynamics");

        Assert.Equal(PromptBuilder.NoInformationAnswer, response.Answer);
        Assert.Equal(0, _chat.CallCount);
    }

    [Fact]
    public async Task Ask_MinimumScore_IsAppliedBeforeReranking()
    {
        await IndexSamplesAsync();

        // The spy re-ranker would reverse the order, but nothing survives the threshold to reach it.
        var response = await CreateService(new RagOptions { MinimumScore = 0.99 })
            .AskAsync("completely unrelated quantum chromodynamics", new QueryOptions(UseReranking: true));

        Assert.Equal(PromptBuilder.NoInformationAnswer, response.Answer);
        Assert.Equal(0, _reranker.CallCount);
    }

    [Fact]
    public async Task Ask_MinimumScore_IsNotAppliedToRankBasedHybridScores()
    {
        await IndexSamplesAsync();

        var response = await CreateService(new RagOptions { MinimumScore = 0.99 })
            .AskAsync("How many vacation days do employees receive per year?", new QueryOptions(UseHybridSearch: true));

        Assert.NotEmpty(response.Sources);
        Assert.Equal(1, _chat.CallCount);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public async Task Ask_EmptyQuestion_Throws(string question)
    {
        await Assert.ThrowsAnyAsync<ArgumentException>(() => CreateService().AskAsync(question));
    }

    [Fact]
    public async Task Ask_InvalidTopK_Throws()
    {
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(
            () => CreateService().AskAsync("q", new QueryOptions(TopK: 0)));
    }

    private sealed class SpyVectorStore : IVectorStore
    {
        private readonly InMemoryVectorStore _inner = new();

        public int SearchCalls { get; private set; }
        public int HybridCalls { get; private set; }
        public int LastTopK { get; private set; }
        public HybridSearchWeights? LastWeights { get; private set; }

        public Task InitializeAsync(CancellationToken ct = default) => _inner.InitializeAsync(ct);
        public Task UpsertAsync(IReadOnlyCollection<VectorRecord> records, CancellationToken ct = default) => _inner.UpsertAsync(records, ct);
        public Task<int> DeleteBySourceAsync(string source, CancellationToken ct = default) => _inner.DeleteBySourceAsync(source, ct);
        public Task ReplaceSourceAsync(string source, IReadOnlyCollection<VectorRecord> records, CancellationToken ct = default) => _inner.ReplaceSourceAsync(source, records, ct);

        public Task<IReadOnlyList<SearchResult>> SearchAsync(ReadOnlyMemory<float> embedding, int topK, CancellationToken ct = default)
        {
            SearchCalls++;
            LastTopK = topK;
            return _inner.SearchAsync(embedding, topK, ct);
        }

        public Task<IReadOnlyList<SearchResult>> HybridSearchAsync(
            ReadOnlyMemory<float> embedding, string text, int topK, HybridSearchWeights weights, CancellationToken ct = default)
        {
            HybridCalls++;
            LastTopK = topK;
            LastWeights = weights;
            return _inner.HybridSearchAsync(embedding, text, topK, weights, ct);
        }
    }

    private sealed class SpyReranker : IReranker
    {
        public int CallCount { get; private set; }
        public int LastTopK { get; private set; }

        public Task<IReadOnlyList<SearchResult>> RerankAsync(
            string question, IReadOnlyList<SearchResult> candidates, int topK, CancellationToken ct = default)
        {
            CallCount++;
            LastTopK = topK;
            IReadOnlyList<SearchResult> reranked = candidates.Reverse().Take(topK).ToList();
            return Task.FromResult(reranked);
        }
    }
}
