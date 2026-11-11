using Rag.Core.Chunking;

namespace Rag.Core.Tests;

public class TextChunkerTests
{
    private static TextChunker Chunker(int size = 2000, int overlap = 200) =>
        new(new ChunkingOptions { ChunkSize = size, ChunkOverlap = overlap });

    private static string Sentences(int count, string prefix = "Sentence") =>
        string.Join(" ", Enumerable.Range(1, count).Select(i => $"{prefix} number {i} has some words in it."));

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("\n\n\n")]
    public void Chunk_EmptyText_ReturnsNoChunks(string text)
    {
        Assert.Empty(Chunker().Chunk("doc", text));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData(" ")]
    public void Chunk_MissingSourceId_Throws(string? sourceId)
    {
        Assert.ThrowsAny<ArgumentException>(() => Chunker().Chunk(sourceId!, "Some text."));
    }

    [Fact]
    public void Chunk_ShortText_ReturnsSingleChunkWithExpectedId()
    {
        var chunks = Chunker().Chunk("notes.md", "Hello world. This is short.");

        var chunk = Assert.Single(chunks);
        Assert.Equal("notes.md-chunk-0", chunk.Id);
        Assert.Equal("notes.md", chunk.SourceId);
        Assert.Equal(0, chunk.Index);
        Assert.Equal("Hello world. This is short.", chunk.Text);
    }

    [Fact]
    public void Chunk_LongText_RespectsChunkSize()
    {
        var chunks = Chunker(size: 300, overlap: 60).Chunk("doc", Sentences(100));

        Assert.True(chunks.Count > 1);
        Assert.All(chunks, c => Assert.InRange(c.Text.Length, 1, 300));
    }

    [Fact]
    public void Chunk_LongText_AssignsSequentialIdsAndIndexes()
    {
        var chunks = Chunker(size: 300, overlap: 60).Chunk("doc", Sentences(50));

        for (var i = 0; i < chunks.Count; i++)
        {
            Assert.Equal(i, chunks[i].Index);
            Assert.Equal($"doc-chunk-{i}", chunks[i].Id);
        }
    }

    [Fact]
    public void Chunk_WithOverlap_NextChunkStartsWithTrailingSentenceOfPreviousChunk()
    {
        var chunks = Chunker(size: 300, overlap: 60).Chunk("doc", Sentences(30));

        Assert.True(chunks.Count > 1);
        for (var i = 1; i < chunks.Count; i++)
        {
            var previousLastSentence = TextChunker.SplitSentences(chunks[i - 1].Text)[^1];
            Assert.StartsWith(previousLastSentence, chunks[i].Text);
        }
    }

    [Fact]
    public void Chunk_WithoutOverlap_DoesNotRepeatSentences()
    {
        var text = Sentences(40);
        var chunks = Chunker(size: 300, overlap: 0).Chunk("doc", text);

        var rejoined = string.Join(" ", chunks.Select(c => c.Text));
        Assert.Equal(text, rejoined);
    }

    [Fact]
    public void Chunk_PreservesEverySentence()
    {
        var text = Sentences(60);
        var chunks = Chunker(size: 250, overlap: 80).Chunk("doc", text);

        foreach (var sentence in TextChunker.SplitSentences(text))
            Assert.Contains(chunks, c => c.Text.Contains(sentence, StringComparison.Ordinal));
    }

    [Fact]
    public void Chunk_SentenceLongerThanChunkSize_IsHardSplit()
    {
        var longSentence = new string('a', 1050);
        var chunks = Chunker(size: 400, overlap: 50).Chunk("doc", longSentence);

        Assert.True(chunks.Count >= 3);
        Assert.All(chunks, c => Assert.True(c.Text.Length <= 400));
        Assert.Equal(1050, chunks[0].Text.Length + chunks.Skip(1).Sum(c => c.Text.Length - 50));
    }

    [Fact]
    public void SplitSentences_SplitsOnSentenceEndingsAndLineBreaks()
    {
        var sentences = TextChunker.SplitSentences("First one. Second one! Third one? Fourth\n\nFifth paragraph");

        Assert.Equal(new[] { "First one.", "Second one!", "Third one?", "Fourth", "Fifth paragraph" }, sentences);
    }

    [Fact]
    public void SplitSentences_DoesNotSplitDecimalNumbers()
    {
        var sentences = TextChunker.SplitSentences("Pi is 3.14 roughly. Next.");

        Assert.Equal(new[] { "Pi is 3.14 roughly.", "Next." }, sentences);
    }

    [Fact]
    public void SplitSentences_KeepsListItemsAndHeadingsAsSeparateSentences()
    {
        var sentences = TextChunker.SplitSentences("# Title\n- first item\n- second item\nClosing line");

        Assert.Equal(new[] { "# Title", "- first item", "- second item", "Closing line" }, sentences);
    }

    [Fact]
    public void Chunk_LongListWithoutFullStops_IsNotCutInsideAnItem()
    {
        var items = Enumerable.Range(1, 80).Select(i => $"- list item number {i} without a full stop").ToList();

        var chunks = Chunker(size: 300, overlap: 0).Chunk("doc", string.Join("\n", items));

        Assert.Equal(string.Join(" ", items), string.Join(" ", chunks.Select(c => c.Text)));
        Assert.All(chunks, c => Assert.StartsWith("- list item number", c.Text));
        Assert.All(chunks, c => Assert.EndsWith("full stop", c.Text));
    }

    [Fact]
    public void SplitSentences_HandlesWindowsLineEndings()
    {
        var sentences = TextChunker.SplitSentences("Para one.\r\n\r\nPara two.");

        Assert.Equal(new[] { "Para one.", "Para two." }, sentences);
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(-1, 0)]
    [InlineData(100, -1)]
    [InlineData(100, 100)]
    [InlineData(100, 150)]
    public void Constructor_InvalidOptions_Throws(int size, int overlap)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Chunker(size, overlap));
    }
}
