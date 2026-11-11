using Rag.Core.Storage;

namespace Rag.Core.Tests;

public class VectorMathTests
{
    [Fact]
    public void CosineSimilarity_IdenticalVectors_IsOne() =>
        Assert.Equal(1.0, VectorMath.CosineSimilarity([1f, 2f, 3f], [1f, 2f, 3f]), precision: 6);

    [Fact]
    public void CosineSimilarity_ScaledVectors_IsOne() =>
        Assert.Equal(1.0, VectorMath.CosineSimilarity([1f, 2f, 3f], [2f, 4f, 6f]), precision: 6);

    [Fact]
    public void CosineSimilarity_OrthogonalVectors_IsZero() =>
        Assert.Equal(0.0, VectorMath.CosineSimilarity([1f, 0f], [0f, 1f]), precision: 6);

    [Fact]
    public void CosineSimilarity_OppositeVectors_IsMinusOne() =>
        Assert.Equal(-1.0, VectorMath.CosineSimilarity([1f, 1f], [-1f, -1f]), precision: 6);

    [Fact]
    public void CosineSimilarity_ZeroVector_IsZero() =>
        Assert.Equal(0.0, VectorMath.CosineSimilarity([0f, 0f], [1f, 1f]));

    [Fact]
    public void CosineSimilarity_DifferentDimensions_Throws() =>
        Assert.Throws<ArgumentException>(() => VectorMath.CosineSimilarity([1f, 2f], [1f, 2f, 3f]));
}

public class HybridSearchWeightsTests
{
    [Fact]
    public void Default_Is70PercentVector30PercentKeyword()
    {
        Assert.Equal(0.7, HybridSearchWeights.Default.VectorWeight);
        Assert.Equal(0.3, HybridSearchWeights.Default.KeywordWeight);
    }

    [Theory]
    [InlineData(1, 1, 1.0)]
    [InlineData(1, null, 0.7)]
    [InlineData(null, 1, 0.3)]
    [InlineData(null, null, 0.0)]
    [InlineData(2, 2, 61.0 / 62.0)]
    public void Fuse_CombinesRanksAndTreatsMissingListsAsZero(int? vectorRank, int? keywordRank, double expected) =>
        Assert.Equal(expected, HybridSearchWeights.Default.Fuse(vectorRank, keywordRank), precision: 6);

    [Fact]
    public void Fuse_BetterRankScoresHigher() =>
        Assert.True(HybridSearchWeights.Default.Fuse(1, null) > HybridSearchWeights.Default.Fuse(2, null));

    [Theory]
    [InlineData(0.0, 0.0)]
    [InlineData(-1.0, 2.0)]
    public void Fuse_InvalidWeights_Throws(double vectorWeight, double keywordWeight) =>
        Assert.Throws<InvalidOperationException>(() => new HybridSearchWeights(vectorWeight, keywordWeight).Fuse(1, 1));
}

public class InMemoryVectorStoreTests
{
    private static VectorRecord Record(string id, string source, float[] embedding, string text = "text", int index = 0) =>
        new(id, source, index, text, embedding);

    [Fact]
    public async Task Search_ReturnsResultsOrderedBySimilarity()
    {
        var store = new InMemoryVectorStore();
        await store.UpsertAsync(
        [
            Record("far", "a", [0f, 1f]),
            Record("close", "a", [1f, 0.1f]),
            Record("middle", "a", [1f, 1f]),
        ]);

        var results = await store.SearchAsync(new float[] { 1f, 0f }, topK: 3);

        Assert.Equal(new[] { "close", "middle", "far" }, results.Select(r => r.Id));
        Assert.True(results[0].Score > results[1].Score);
    }

    [Fact]
    public async Task Search_RespectsTopK()
    {
        var store = new InMemoryVectorStore();
        await store.UpsertAsync(Enumerable.Range(0, 10).Select(i => Record($"r{i}", "a", [1f, i])).ToList());

        var results = await store.SearchAsync(new float[] { 1f, 0f }, topK: 4);

        Assert.Equal(4, results.Count);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-3)]
    public async Task Search_InvalidTopK_Throws(int topK)
    {
        var store = new InMemoryVectorStore();
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => store.SearchAsync(new float[] { 1f }, topK));
    }

    [Fact]
    public async Task Search_EmptyStore_ReturnsEmpty()
    {
        var results = await new InMemoryVectorStore().SearchAsync(new float[] { 1f, 0f }, 5);
        Assert.Empty(results);
    }

    [Fact]
    public async Task Upsert_SameId_ReplacesRecord()
    {
        var store = new InMemoryVectorStore();
        await store.UpsertAsync([Record("x", "a", [1f, 0f], "old")]);
        await store.UpsertAsync([Record("x", "a", [1f, 0f], "new")]);

        var result = Assert.Single(await store.SearchAsync(new float[] { 1f, 0f }, 5));
        Assert.Equal("new", result.Text);
        Assert.Equal(1, store.Count);
    }

    [Fact]
    public async Task DeleteBySource_RemovesOnlyThatSource()
    {
        var store = new InMemoryVectorStore();
        await store.UpsertAsync(
        [
            Record("a-0", "a", [1f, 0f]),
            Record("a-1", "a", [1f, 0f]),
            Record("b-0", "b", [1f, 0f]),
        ]);

        var removed = await store.DeleteBySourceAsync("a");

        Assert.Equal(2, removed);
        var remaining = Assert.Single(await store.SearchAsync(new float[] { 1f, 0f }, 5));
        Assert.Equal("b", remaining.Source);
    }

    [Fact]
    public async Task ReplaceSource_SwapsAllChunksOfThatSourceOnly()
    {
        var store = new InMemoryVectorStore();
        await store.UpsertAsync(
        [
            Record("a-0", "a", [1f, 0f], "old zero"),
            Record("a-1", "a", [1f, 0f], "old one"),
            Record("b-0", "b", [0f, 1f], "other"),
        ]);

        await store.ReplaceSourceAsync("a", [Record("a-0", "a", [1f, 0f], "new zero")]);

        var results = await store.SearchAsync(new float[] { 1f, 0f }, 10);
        Assert.Equal(new[] { "new zero", "other" }, results.Select(r => r.Text));
    }

    [Fact]
    public async Task HybridSearch_KeywordMatchCanOutrankSlightlyCloserVector()
    {
        var store = new InMemoryVectorStore();
        await store.UpsertAsync(
        [
            Record("semantic", "a", [1f, 0f], "Annual leave policy overview"),
            Record("keyword", "b", [0.9f, 0.3f], "How many vacation days do employees get"),
        ]);

        var vectorOnly = await store.SearchAsync(new float[] { 1f, 0f }, 2);
        var hybrid = await store.HybridSearchAsync(new float[] { 1f, 0f }, "vacation days", 2, HybridSearchWeights.Default);

        Assert.Equal("semantic", vectorOnly[0].Id);
        Assert.Equal("keyword", hybrid[0].Id);
    }

    [Theory]
    [InlineData("vacation days", "Employees get vacation days.", 1.0)]
    [InlineData("vacation days", "Employees get vacation.", 0.5)]
    [InlineData("vacation days", "Nothing relevant here.", 0.0)]
    [InlineData("VACATION", "vacation is fun", 1.0)]
    public void KeywordScore_IsFractionOfMatchedQueryTerms(string query, string text, double expected)
    {
        var terms = query.Split(' ').ToHashSet(StringComparer.OrdinalIgnoreCase);
        Assert.Equal(expected, InMemoryVectorStore.KeywordScore(terms, text), precision: 6);
    }
}
