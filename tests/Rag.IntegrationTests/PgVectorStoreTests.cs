using Npgsql;
using Rag.Core.Chunking;
using Rag.Core.Indexing;
using Rag.Core.Storage;
using Rag.Infrastructure;
using Rag.Testing;
using Testcontainers.PostgreSql;
using Rag.Core.Security;

namespace Rag.IntegrationTests;

/// <summary>Starts one pgvector container for all tests in the class. Requires Docker.</summary>
public sealed class PostgresFixture : IAsyncLifetime
{
    private readonly PostgreSqlContainer _container = new PostgreSqlBuilder()
        .WithImage("pgvector/pgvector:pg17")
        .Build();

    public NpgsqlDataSource DataSource { get; private set; } = null!;

    public async Task InitializeAsync()
    {
        await _container.StartAsync();
        DataSource = PgVectorDataSource.Create(_container.GetConnectionString());
    }

    public async Task DisposeAsync()
    {
        await DataSource.DisposeAsync();
        await _container.DisposeAsync();
    }
}

[Trait("Category", "Integration")]
public class PgVectorStoreTests : IClassFixture<PostgresFixture>
{
    private const int Dimensions = 64;
    private readonly PostgresFixture _fixture;

    public PgVectorStoreTests(PostgresFixture fixture) => _fixture = fixture;

    /// <summary>Each test gets its own table so tests stay independent.</summary>
    private static string NewTableName() => $"emb_{Guid.NewGuid():N}"[..16];

    private async Task<PgVectorStore> CreateStoreAsync(string? tableName = null)
    {
        var options = new PgVectorOptions
        {
            Dimensions = Dimensions,
            TableName = tableName ?? NewTableName(),
        };
        var store = new PgVectorStore(_fixture.DataSource, options);
        await store.InitializeAsync();
        return store;
    }

    private static float[] UnitVector(int hotIndex, float noise = 0f)
    {
        var v = new float[Dimensions];
        v[hotIndex] = 1f;
        if (noise != 0f)
            v[(hotIndex + 1) % Dimensions] = noise;
        return v;
    }

    [Fact]
    public async Task Initialize_IsIdempotent()
    {
        var store = await CreateStoreAsync();

        await store.InitializeAsync();
        await store.InitializeAsync();
    }

    [Fact]
    public async Task Search_ReturnsNearestByCosineSimilarity()
    {
        var store = await CreateStoreAsync();
        await store.UpsertAsync(
        [
            new VectorRecord("exact", "doc", 0, "exact match", UnitVector(3)),
            new VectorRecord("near", "doc", 1, "near match", UnitVector(3, noise: 0.5f)),
            new VectorRecord("far", "doc", 2, "far away", UnitVector(10)),
        ]);

        var results = await store.SearchAsync(UnitVector(3), topK: 2, scope: AccessScope.Unrestricted);

        Assert.Equal(new[] { "exact", "near" }, results.Select(r => r.Id));
        Assert.Equal(1.0, results[0].Score, precision: 4);
        Assert.True(results[1].Score < results[0].Score);
    }

    [Fact]
    public async Task Upsert_ExistingId_UpdatesTextAndEmbedding()
    {
        var store = await CreateStoreAsync();
        await store.UpsertAsync([new VectorRecord("x", "doc", 0, "old", UnitVector(1))]);
        await store.UpsertAsync([new VectorRecord("x", "doc", 0, "new", UnitVector(2))]);

        var result = Assert.Single(await store.SearchAsync(UnitVector(2), topK: 5, scope: AccessScope.Unrestricted));
        Assert.Equal("new", result.Text);
        Assert.Equal(1.0, result.Score, precision: 4);
    }

    [Fact]
    public async Task DeleteBySource_RemovesOnlyThatSource()
    {
        var store = await CreateStoreAsync();
        await store.UpsertAsync(
        [
            new VectorRecord("a-0", "a", 0, "a zero", UnitVector(1)),
            new VectorRecord("a-1", "a", 1, "a one", UnitVector(2)),
            new VectorRecord("b-0", "b", 0, "b zero", UnitVector(3)),
        ]);

        var removed = await store.DeleteBySourceAsync("a", AccessScope.Unrestricted);

        Assert.Equal(2, removed);
        var remaining = Assert.Single(await store.SearchAsync(UnitVector(1), topK: 10, scope: AccessScope.Unrestricted));
        Assert.Equal("b", remaining.Source);
    }

    [Fact]
    public async Task ReplaceSource_SwapsChunksAndLeavesOtherSourcesAlone()
    {
        var store = await CreateStoreAsync();
        await store.UpsertAsync(
        [
            new VectorRecord("a-0", "a", 0, "old zero", UnitVector(1)),
            new VectorRecord("a-1", "a", 1, "old one", UnitVector(2)),
            new VectorRecord("b-0", "b", 0, "other", UnitVector(3)),
        ]);

        await store.ReplaceSourceAsync("a", [new VectorRecord("a-0", "a", 0, "new zero", UnitVector(1))], AccessScope.Unrestricted);

        var results = await store.SearchAsync(UnitVector(1), topK: 10, scope: AccessScope.Unrestricted);
        Assert.Equal(new[] { "new zero", "other" }, results.Select(r => r.Text).Order());
    }

    [Fact]
    public async Task ReplaceSource_WhenInsertFails_KeepsTheOldChunks()
    {
        var store = await CreateStoreAsync();
        await store.UpsertAsync([new VectorRecord("a-0", "a", 0, "original", UnitVector(1))]);

        // Wrong vector size makes the insert fail after the delete ran inside the same transaction.
        var wrongSize = new VectorRecord("a-0", "a", 0, "replacement", new float[Dimensions + 1]);
        await Assert.ThrowsAnyAsync<NpgsqlException>(() => store.ReplaceSourceAsync("a", [wrongSize], AccessScope.Unrestricted));

        var survivor = Assert.Single(await store.SearchAsync(UnitVector(1), topK: 5, scope: AccessScope.Unrestricted));
        Assert.Equal("original", survivor.Text);
    }

    [Fact]
    public async Task Search_MoreThanDefaultHnswCandidates_ReturnsAllRequestedRows()
    {
        // HNSW returns at most hnsw.ef_search rows (40 by default); the store must raise it.
        var store = await CreateStoreAsync();
        var records = Enumerable.Range(0, 60)
            .Select(i => new VectorRecord($"r{i}", "doc", i, $"row {i}", UnitVector(i % Dimensions, noise: 0.01f * (i + 1))))
            .ToList();
        await store.UpsertAsync(records);

        var results = await store.SearchAsync(UnitVector(0, noise: 0.5f), topK: 55, scope: AccessScope.Unrestricted);

        Assert.Equal(55, results.Count);
    }

    [Fact]
    public async Task Initialize_ExistingTableWithOtherDimensions_ThrowsClearError()
    {
        var tableName = NewTableName();
        await CreateStoreAsync(tableName);

        var mismatched = new PgVectorStore(_fixture.DataSource, new PgVectorOptions { Dimensions = Dimensions / 2, TableName = tableName });

        var error = await Assert.ThrowsAsync<InvalidOperationException>(() => mismatched.InitializeAsync());
        Assert.Contains($"{Dimensions}-dimensional", error.Message);
    }

    [Fact]
    public async Task HybridSearch_BoostsChunksMatchingQueryKeywords()
    {
        var store = await CreateStoreAsync();
        await store.UpsertAsync(
        [
            new VectorRecord("semantic", "doc", 0, "Annual leave policy overview", UnitVector(5)),
            new VectorRecord("keyword", "doc", 1, "Employees get twenty five vacation days", UnitVector(5, noise: 0.2f)),
        ]);

        var vectorOnly = await store.SearchAsync(UnitVector(5), topK: 2, scope: AccessScope.Unrestricted);
        var hybrid = await store.HybridSearchAsync(UnitVector(5), "vacation days", topK: 2, new HybridSearchWeights(0.5, 0.5), scope: AccessScope.Unrestricted);

        Assert.Equal("semantic", vectorOnly[0].Id);
        Assert.Equal("keyword", hybrid[0].Id);
    }

    [Fact]
    public async Task HybridSearch_NoKeywordMatches_FallsBackToVectorResults()
    {
        var store = await CreateStoreAsync();
        await store.UpsertAsync([new VectorRecord("only", "doc", 0, "nothing in common", UnitVector(7))]);

        var results = await store.HybridSearchAsync(UnitVector(7), "zebra", topK: 3, HybridSearchWeights.Default, scope: AccessScope.Unrestricted);

        var result = Assert.Single(results);
        // Rank 1 in the vector list only: the vector weight, on the 0..1 fused scale.
        Assert.Equal(0.7, result.Score, precision: 4);
    }

    [Fact]
    public async Task IndexerAndStore_EndToEnd_FindsRelevantChunk()
    {
        var store = await CreateStoreAsync();
        var embeddings = new FakeEmbeddingGenerator(Dimensions);
        var indexer = new DocumentIndexer(
            new TextChunker(new ChunkingOptions { ChunkSize = 120, ChunkOverlap = 0 }), embeddings, store);

        await indexer.IndexDocumentAsync(SampleDocuments.HandbookName, SampleDocuments.Handbook, AccessScope.Unrestricted);
        await indexer.IndexDocumentAsync(SampleDocuments.RecipesName, SampleDocuments.Recipes, AccessScope.Unrestricted);

        var query = embeddings.Embed("How long do I cook the spaghetti pasta?");
        var results = await store.SearchAsync(query, topK: 1, scope: AccessScope.Unrestricted);

        Assert.Equal(SampleDocuments.RecipesName, Assert.Single(results).Source);
    }

    [Fact]
    public void Constructor_RejectsUnsafeTableName()
    {
        Assert.Throws<ArgumentException>(() =>
            new PgVectorStore(_fixture.DataSource, new PgVectorOptions { TableName = "x; DROP TABLE users" }));
    }
}
