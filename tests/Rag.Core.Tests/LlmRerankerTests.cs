using Rag.Core.Retrieval;
using Rag.Core.Storage;
using Rag.Testing;

namespace Rag.Core.Tests;

public class LlmRerankerTests
{
    private static List<SearchResult> Candidates(int count) =>
        Enumerable.Range(0, count)
            .Select(i => new SearchResult($"id{i}", "doc", i, $"Candidate text {i}", 0.5))
            .ToList();

    [Fact]
    public async Task Rerank_ReordersByModelScores()
    {
        var chat = new FakeChatClient("[0.1, 0.9, 0.5]");
        var reranker = new LlmReranker(chat);

        var result = await reranker.RerankAsync("question", Candidates(3), topK: 3);

        Assert.Equal(new[] { "id1", "id2", "id0" }, result.Select(r => r.Id));
        Assert.Equal(new[] { 0.9, 0.5, 0.1 }, result.Select(r => r.Score));
    }

    [Fact]
    public async Task Rerank_TrimsToTopK()
    {
        var reranker = new LlmReranker(new FakeChatClient("[0.2, 0.8, 0.4, 0.6]"));

        var result = await reranker.RerankAsync("q", Candidates(4), topK: 2);

        Assert.Equal(new[] { "id1", "id3" }, result.Select(r => r.Id));
    }

    [Fact]
    public async Task Rerank_UnusableResponse_KeepsOriginalOrder()
    {
        var reranker = new LlmReranker(new FakeChatClient("Sorry, I cannot help with that."));

        var result = await reranker.RerankAsync("q", Candidates(3), topK: 2);

        Assert.Equal(new[] { "id0", "id1" }, result.Select(r => r.Id));
    }

    [Fact]
    public async Task Rerank_SingleCandidate_DoesNotCallModel()
    {
        var chat = new FakeChatClient("[1.0]");

        var result = await new LlmReranker(chat).RerankAsync("q", Candidates(1), topK: 5);

        Assert.Single(result);
        Assert.Equal(0, chat.CallCount);
    }

    [Fact]
    public async Task Rerank_PromptContainsQuestionAndTruncatedCandidates()
    {
        var chat = new FakeChatClient("[0.5, 0.5]");
        var longText = new string('x', 500);
        var candidates = new List<SearchResult>
        {
            new("a", "doc", 0, longText, 0.5),
            new("b", "doc", 1, "short", 0.5),
        };

        await new LlmReranker(chat).RerankAsync("What is X?", candidates, topK: 2);

        var prompt = Assert.Single(chat.LastRequest).Text;
        Assert.Contains("What is X?", prompt);
        Assert.Contains(new string('x', LlmReranker.PreviewLength), prompt);
        Assert.DoesNotContain(new string('x', LlmReranker.PreviewLength + 1), prompt);
        Assert.Contains("Document 2: short", prompt);
    }

    [Theory]
    [InlineData("[0.1, 0.2]", 2, new[] { 0.1, 0.2 })]
    [InlineData("```json\n[0.3, 0.7]\n```", 2, new[] { 0.3, 0.7 })]
    [InlineData("Scores: [1, 0] done", 2, new[] { 1.0, 0.0 })]
    [InlineData("[\"0.4\", \"0.6\"]", 2, new[] { 0.4, 0.6 })]
    [InlineData("[1.7, -0.5]", 2, new[] { 1.0, 0.0 })]
    public void ParseScores_ValidInput_ReturnsClampedScores(string text, int count, double[] expected)
    {
        Assert.Equal(expected, LlmReranker.ParseScores(text, count));
    }

    [Theory]
    [InlineData(null, 2)]
    [InlineData("", 2)]
    [InlineData("no array here", 2)]
    [InlineData("[0.1, 0.2, 0.3]", 2)]
    [InlineData("[0.1, \"high\"]", 2)]
    [InlineData("[0.1, ", 2)]
    [InlineData("[{\"score\": 1}, 0.2]", 2)]
    public void ParseScores_InvalidInput_ReturnsNull(string? text, int count)
    {
        Assert.Null(LlmReranker.ParseScores(text, count));
    }
}
