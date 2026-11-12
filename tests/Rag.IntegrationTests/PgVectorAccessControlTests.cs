using Rag.Core.Security;
using Rag.Core.Storage;
using Rag.Infrastructure;

namespace Rag.IntegrationTests;

/// <summary>Access-control behaviour of the Postgres store: the filter runs inside SQL.</summary>
[Trait("Category", "Integration")]
public class PgVectorAccessControlTests : IClassFixture<PostgresFixture>
{
    private const int Dimensions = 64;
    private static readonly AccessScope Alice = AccessScope.ForUser("alice");
    private static readonly AccessScope Bob = AccessScope.ForUser("bob");
    private static readonly AccessScope HrMember = AccessScope.ForUser("carol", ["hr"]);

    private readonly PostgresFixture _fixture;

    public PgVectorAccessControlTests(PostgresFixture fixture) => _fixture = fixture;

    private static string NewTableName() => $"acl_{Guid.NewGuid():N}"[..16];

    private async Task<PgVectorStore> CreateStoreAsync(string? tableName = null)
    {
        var store = new PgVectorStore(_fixture.DataSource, new PgVectorOptions
        {
            Dimensions = Dimensions,
            TableName = tableName ?? NewTableName(),
        });
        await store.InitializeAsync();
        return store;
    }

    private static float[] UnitVector(int hotIndex)
    {
        var v = new float[Dimensions];
        v[hotIndex] = 1f;
        return v;
    }

    private static VectorRecord Rec(string id, string source, string owner, int hot, string text, params string[] acl) =>
        new(id, source, 0, text, UnitVector(hot), owner, acl);

    [Fact]
    public async Task Search_ReturnsOnlyChunksTheCallerMayRead()
    {
        var store = await CreateStoreAsync();
        await store.UpsertAsync(
        [
            Rec("a", "alice-doc", "user:alice", 0, "alpha", "user:alice"),
            Rec("b", "bob-doc", "user:bob", 0, "beta", "user:bob"),
            Rec("h", "hr-doc", "user:bob", 0, "gamma", "user:bob", "group:hr"),
        ]);

        var forAlice = await store.SearchAsync(UnitVector(0), 10, Alice);
        var forHr = await store.SearchAsync(UnitVector(0), 10, HrMember);
        var forAll = await store.SearchAsync(UnitVector(0), 10, AccessScope.Unrestricted);

        Assert.Equal(new[] { "a" }, forAlice.Select(r => r.Id));
        Assert.Equal(new[] { "h" }, forHr.Select(r => r.Id));
        Assert.Equal(3, forAll.Count);
    }

    [Fact]
    public async Task HybridSearch_ReturnsOnlyChunksTheCallerMayRead()
    {
        var store = await CreateStoreAsync();
        await store.UpsertAsync(
        [
            Rec("a", "alice-doc", "user:alice", 0, "vacation days for alice", "user:alice"),
            Rec("b", "bob-doc", "user:bob", 1, "vacation days for bob", "user:bob"),
        ]);

        var forBob = await store.HybridSearchAsync(UnitVector(0), "vacation days", 10, HybridSearchWeights.Default, Bob);

        Assert.Equal(new[] { "b" }, forBob.Select(r => r.Id));
    }

    [Fact]
    public async Task Search_WhenMostNearestChunksAreForbidden_StillFindsTheReadableOne()
    {
        // 60 chunks nearer to the query than alice's own one, all owned by bob.
        var store = await CreateStoreAsync();
        var records = Enumerable.Range(0, 60)
            .Select(i => Rec($"bob-{i}", "bob-doc", "user:bob", 0, $"bob {i}", "user:bob"))
            .Append(Rec("alice-0", "alice-doc", "user:alice", 1, "alice", "user:alice"))
            .ToList();
        await store.UpsertAsync(records);

        var results = await store.SearchAsync(UnitVector(0), 5, Alice);

        Assert.Equal(new[] { "alice-0" }, results.Select(r => r.Id));
    }

    [Fact]
    public async Task ReplaceSource_ByAnotherOwner_ThrowsAndKeepsTheDocument()
    {
        var store = await CreateStoreAsync();
        await store.ReplaceSourceAsync("doc", [Rec("a-0", "doc", "user:alice", 0, "alice text", "user:alice")], Alice);

        await Assert.ThrowsAsync<SourceOwnershipException>(() =>
            store.ReplaceSourceAsync("doc", [Rec("b-0", "doc", "user:bob", 0, "bob text", "user:bob")], Bob));

        var survivor = Assert.Single(await store.SearchAsync(UnitVector(0), 10, Alice));
        Assert.Equal("a-0", survivor.Id);
        Assert.Empty(await store.SearchAsync(UnitVector(0), 10, Bob));
    }

    [Fact]
    public async Task ReplaceSource_ByTheOwner_CanChangeSharing()
    {
        var store = await CreateStoreAsync();
        await store.ReplaceSourceAsync("doc", [Rec("v1", "doc", "user:alice", 0, "text", "user:alice")], Alice);

        await store.ReplaceSourceAsync("doc", [Rec("v2", "doc", "user:alice", 0, "text", "user:alice", "user:bob")], Alice);

        Assert.Equal(new[] { "v2" }, (await store.SearchAsync(UnitVector(0), 10, Bob)).Select(r => r.Id));
    }

    [Fact]
    public async Task DeleteBySource_ByAnotherOwner_Throws_ByOwnerSucceeds()
    {
        var store = await CreateStoreAsync();
        await store.ReplaceSourceAsync("doc", [Rec("a-0", "doc", "user:alice", 0, "text", "user:alice", "user:bob")], Alice);

        await Assert.ThrowsAsync<SourceOwnershipException>(() => store.DeleteBySourceAsync("doc", Bob));
        Assert.Equal(1, await store.DeleteBySourceAsync("doc", Alice));
        Assert.Empty(await store.SearchAsync(UnitVector(0), 10, AccessScope.Unrestricted));
    }

    [Fact]
    public async Task ConcurrentFirstUploads_OfTheSameDocument_LetExactlyOneOwnerWin()
    {
        var store = await CreateStoreAsync();

        var attempts = await Task.WhenAll(
            Attempt(store, Alice, "user:alice"),
            Attempt(store, Bob, "user:bob"));

        Assert.Equal(1, attempts.Count(ok => ok));
    }

    private static async Task<bool> Attempt(PgVectorStore store, AccessScope scope, string owner)
    {
        try
        {
            await store.ReplaceSourceAsync("doc", [Rec($"{owner}-0", "doc", owner, 0, "text", owner)], scope);
            return true;
        }
        catch (SourceOwnershipException)
        {
            return false;
        }
    }

    [Fact]
    public async Task Initialize_OnATableWithoutAccessColumns_MigratesAndHidesOldRows()
    {
        var table = NewTableName();
        var store = await CreateStoreAsync(table);
        await store.UpsertAsync([Rec("old", "old-doc", AccessScope.SystemOwnerPrincipal, 0, "legacy")]);

        // Simulate a table created before access control existed.
        await using (var cmd = _fixture.DataSource.CreateCommand($"ALTER TABLE \"{table}\" DROP COLUMN owner, DROP COLUMN acl"))
            await cmd.ExecuteNonQueryAsync();

        await store.InitializeAsync();

        Assert.Empty(await store.SearchAsync(UnitVector(0), 10, Alice));
        Assert.Single(await store.SearchAsync(UnitVector(0), 10, AccessScope.Unrestricted));
    }
}
