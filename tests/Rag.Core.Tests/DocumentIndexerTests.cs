using Rag.Core.Chunking;
using Rag.Core.Indexing;
using Rag.Core.Storage;
using Rag.Testing;
using Rag.Core.Security;

namespace Rag.Core.Tests;

public class DocumentIndexerTests
{
    private readonly FakeEmbeddingGenerator _embeddings = new();
    private readonly InMemoryVectorStore _store = new();

    private DocumentIndexer CreateIndexer(int chunkSize = 2000, int overlap = 200) =>
        new(new TextChunker(new ChunkingOptions { ChunkSize = chunkSize, ChunkOverlap = overlap }), _embeddings, _store);

    private static string ManySentences(int count) =>
        string.Join(" ", Enumerable.Range(1, count).Select(i => $"Fact {i} is stored here."));

    [Fact]
    public async Task IndexDocument_StoresEveryChunkWithItsEmbedding()
    {
        var result = await CreateIndexer(chunkSize: 100, overlap: 0).IndexDocumentAsync("doc.txt", ManySentences(20), AccessScope.Unrestricted);

        Assert.Equal("doc.txt", result.DocumentId);
        Assert.True(result.ChunkCount > 1);
        Assert.Equal(result.ChunkCount, _store.Count);

        var hits = await _store.SearchAsync(_embeddings.Embed("Fact 7 is stored here."), topK: result.ChunkCount, scope: AccessScope.Unrestricted);
        Assert.All(hits, h => Assert.Equal("doc.txt", h.Source));
        Assert.Contains(hits, h => h.Text.Contains("Fact 7 is stored here."));
    }

    [Fact]
    public async Task IndexDocument_EmbedsChunksInBatches()
    {
        // ~70 sentences of ~24 chars with a 30-char chunk size → one chunk per sentence.
        var result = await CreateIndexer(chunkSize: 30, overlap: 0).IndexDocumentAsync("doc", ManySentences(DocumentIndexer.EmbeddingBatchSize + 6), AccessScope.Unrestricted);

        Assert.Equal(DocumentIndexer.EmbeddingBatchSize + 6, result.ChunkCount);
        Assert.Equal(2, _embeddings.CallCount);
        Assert.Equal(result.ChunkCount, _embeddings.Inputs.Count);
    }

    [Fact]
    public async Task IndexDocument_ReindexingRemovesStaleChunks()
    {
        var indexer = CreateIndexer(chunkSize: 100, overlap: 0);
        var first = await indexer.IndexDocumentAsync("doc", ManySentences(30), AccessScope.Unrestricted);
        var second = await indexer.IndexDocumentAsync("doc", "Only one short sentence now.", AccessScope.Unrestricted);

        Assert.True(first.ChunkCount > 1);
        Assert.Equal(1, second.ChunkCount);
        Assert.Equal(1, _store.Count);
    }

    [Fact]
    public async Task IndexDocument_FailedReindex_KeepsThePreviousVersion()
    {
        var indexer = CreateIndexer();
        await indexer.IndexDocumentAsync("doc", "The original content stays.", AccessScope.Unrestricted);

        _embeddings.ForcedResultCount = 0;
        await Assert.ThrowsAsync<InvalidOperationException>(() => indexer.IndexDocumentAsync("doc", "Replacement content.", AccessScope.Unrestricted));

        var hit = Assert.Single(await _store.SearchAsync(_embeddings.Embed("original content"), topK: 5, scope: AccessScope.Unrestricted));
        Assert.Equal("The original content stays.", hit.Text);
    }

    [Fact]
    public async Task IndexDocument_DoesNotTouchOtherDocuments()
    {
        var indexer = CreateIndexer();
        await indexer.IndexDocumentAsync(SampleDocuments.HandbookName, SampleDocuments.Handbook, AccessScope.Unrestricted);
        await indexer.IndexDocumentAsync(SampleDocuments.RecipesName, SampleDocuments.Recipes, AccessScope.Unrestricted);
        await indexer.IndexDocumentAsync(SampleDocuments.RecipesName, SampleDocuments.Recipes, AccessScope.Unrestricted);

        Assert.Equal(2, _store.Count);
    }

    [Fact]
    public async Task IndexDocument_EmbeddingCountMismatch_Throws()
    {
        _embeddings.ForcedResultCount = 0;

        await Assert.ThrowsAsync<InvalidOperationException>(
            () => CreateIndexer().IndexDocumentAsync("doc", "Some content.", AccessScope.Unrestricted));
        Assert.Equal(0, _store.Count);
    }

    [Theory]
    [InlineData("", "content")]
    [InlineData("doc", "")]
    [InlineData("doc", "   ")]
    public async Task IndexDocument_MissingInput_Throws(string documentId, string content)
    {
        await Assert.ThrowsAnyAsync<ArgumentException>(() => CreateIndexer().IndexDocumentAsync(documentId, content, AccessScope.Unrestricted));
    }
}
