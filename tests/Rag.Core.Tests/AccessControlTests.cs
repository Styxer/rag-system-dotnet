using Rag.Core.Chunking;
using Rag.Core.Indexing;
using Rag.Core.Security;
using Rag.Core.Storage;
using Rag.Testing;

namespace Rag.Core.Tests;

public class AccessControlTests
{
    private static readonly AccessScope Alice = AccessScope.ForUser("alice");
    private static readonly AccessScope Bob = AccessScope.ForUser("bob");
    private static readonly AccessScope HrMember = AccessScope.ForUser("carol", ["hr"]);
    private static readonly float[] Query = [1f, 0f];

    private static VectorRecord Rec(string id, string source, string owner, params string[] acl) =>
        new(id, source, 0, "vacation days policy", new float[] { 1f, 0f }, owner, acl);

    [Fact]
    public async Task Search_ReturnsOnlyChunksTheCallerMayRead()
    {
        var store = new InMemoryVectorStore();
        await store.UpsertAsync(
        [
            Rec("a", "alice-doc", "user:alice", "user:alice"),
            Rec("b", "bob-doc", "user:bob", "user:bob"),
            Rec("h", "hr-doc", "user:bob", "user:bob", "group:hr"),
        ]);

        var forAlice = await store.SearchAsync(Query, 10, Alice);
        var forHr = await store.SearchAsync(Query, 10, HrMember);

        Assert.Equal(new[] { "a" }, forAlice.Select(r => r.Id));
        Assert.Equal(new[] { "h" }, forHr.Select(r => r.Id));
    }

    [Fact]
    public async Task HybridSearch_ReturnsOnlyChunksTheCallerMayRead()
    {
        var store = new InMemoryVectorStore();
        await store.UpsertAsync(
        [
            Rec("a", "alice-doc", "user:alice", "user:alice"),
            Rec("b", "bob-doc", "user:bob", "user:bob"),
        ]);

        var forBob = await store.HybridSearchAsync(Query, "vacation days", 10, HybridSearchWeights.Default, Bob);

        Assert.Equal(new[] { "b" }, forBob.Select(r => r.Id));
    }

    [Fact]
    public async Task Chunks_WithoutAnAccessList_AreInvisibleToUsers_ButVisibleToUnrestricted()
    {
        var store = new InMemoryVectorStore();
        await store.UpsertAsync([new VectorRecord("legacy", "old", 0, "text", new float[] { 1f, 0f })]);

        Assert.Empty(await store.SearchAsync(Query, 10, Alice));
        Assert.Single(await store.SearchAsync(Query, 10, AccessScope.Unrestricted));
    }

    [Fact]
    public async Task ReplaceSource_ByAnotherOwner_ThrowsAndKeepsTheDocument()
    {
        var store = new InMemoryVectorStore();
        await store.ReplaceSourceAsync("doc", [Rec("a-0", "doc", "user:alice", "user:alice")], Alice);

        var ex = await Assert.ThrowsAsync<SourceOwnershipException>(() =>
            store.ReplaceSourceAsync("doc", [Rec("b-0", "doc", "user:bob", "user:bob")], Bob));

        Assert.Equal("doc", ex.Source);
        Assert.Equal(new[] { "a-0" }, (await store.SearchAsync(Query, 10, Alice)).Select(r => r.Id));
        Assert.Empty(await store.SearchAsync(Query, 10, Bob));
    }

    [Fact]
    public async Task ReplaceSource_ByTheOwner_Replaces()
    {
        var store = new InMemoryVectorStore();
        await store.ReplaceSourceAsync("doc", [Rec("v1", "doc", "user:alice", "user:alice")], Alice);

        await store.ReplaceSourceAsync("doc", [Rec("v2", "doc", "user:alice", "user:alice", "user:bob")], Alice);

        Assert.Equal(new[] { "v2" }, (await store.SearchAsync(Query, 10, Bob)).Select(r => r.Id));
        Assert.Equal(1, store.Count);
    }

    [Fact]
    public async Task ReplaceSource_RecordsMustNameTheCallerAndTheDocument()
    {
        var store = new InMemoryVectorStore();

        await Assert.ThrowsAsync<ArgumentException>(() =>
            store.ReplaceSourceAsync("doc", [Rec("x", "doc", "user:bob", "user:bob")], Alice));
        await Assert.ThrowsAsync<ArgumentException>(() =>
            store.ReplaceSourceAsync("doc", [Rec("x", "other-doc", "user:alice", "user:alice")], Alice));
        Assert.Equal(0, store.Count);
    }

    [Fact]
    public async Task DeleteBySource_ByAnotherOwner_Throws_ByOwnerOrUnrestrictedSucceeds()
    {
        var store = new InMemoryVectorStore();
        await store.ReplaceSourceAsync("doc", [Rec("a-0", "doc", "user:alice", "user:alice", "user:bob")], Alice);

        await Assert.ThrowsAsync<SourceOwnershipException>(() => store.DeleteBySourceAsync("doc", Bob));
        Assert.Equal(1, store.Count);

        Assert.Equal(1, await store.DeleteBySourceAsync("doc", Alice));

        await store.ReplaceSourceAsync("doc", [Rec("a-1", "doc", "user:alice", "user:alice")], Alice);
        Assert.Equal(1, await store.DeleteBySourceAsync("doc", AccessScope.Unrestricted));
    }

    [Fact]
    public async Task DeleteBySource_UnknownDocument_RemovesNothing()
    {
        var store = new InMemoryVectorStore();

        Assert.Equal(0, await store.DeleteBySourceAsync("missing", Alice));
    }
}

public class IndexerAccessTests
{
    private static readonly AccessScope Alice = AccessScope.ForUser("alice");
    private static readonly AccessScope Bob = AccessScope.ForUser("bob");
    private static readonly AccessScope HrMember = AccessScope.ForUser("carol", ["hr"]);

    private readonly FakeEmbeddingGenerator _embeddings = new();
    private readonly InMemoryVectorStore _store = new();

    private DocumentIndexer CreateIndexer() =>
        new(new TextChunker(new ChunkingOptions { ChunkSize = 2000, ChunkOverlap = 0 }), _embeddings, _store);

    private Task<IReadOnlyList<SearchResult>> SearchAs(AccessScope scope) =>
        _store.SearchAsync(_embeddings.Embed("Employees get 25 vacation days."), 10, scope);

    [Fact]
    public async Task IndexedDocument_IsPrivateToItsUploader_ByDefault()
    {
        await CreateIndexer().IndexDocumentAsync("doc", "Employees get 25 vacation days.", Alice);

        Assert.NotEmpty(await SearchAs(Alice));
        Assert.Empty(await SearchAs(Bob));
        Assert.Empty(await SearchAs(HrMember));
    }

    [Fact]
    public async Task IndexedDocument_CanBeSharedWithUsersAndGroups()
    {
        await CreateIndexer().IndexDocumentAsync("doc", "Employees get 25 vacation days.", Alice, ["group:hr", "user:bob"]);

        Assert.NotEmpty(await SearchAs(Bob));
        Assert.NotEmpty(await SearchAs(HrMember));
        Assert.Empty(await SearchAs(AccessScope.ForUser("dave")));
    }

    [Fact]
    public async Task ReindexingSomeoneElsesDocument_Throws_AndKeepsTheOriginal()
    {
        await CreateIndexer().IndexDocumentAsync("doc", "Employees get 25 vacation days.", Alice);

        await Assert.ThrowsAsync<SourceOwnershipException>(() =>
            CreateIndexer().IndexDocumentAsync("doc", "Replaced by bob.", Bob));

        var hit = Assert.Single(await SearchAs(Alice));
        Assert.Contains("vacation days", hit.Text);
    }

    [Fact]
    public async Task InvalidSharingPrincipal_Throws_AndStoresNothing()
    {
        await Assert.ThrowsAsync<ArgumentException>(() =>
            CreateIndexer().IndexDocumentAsync("doc", "Some content.", Alice, ["everyone"]));

        Assert.Equal(0, _store.Count);
    }
}
