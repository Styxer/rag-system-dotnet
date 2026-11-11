namespace Rag.Core.Chunking;

public sealed class ChunkingOptions
{
    public const string SectionName = "Chunking";

    /// <summary>Maximum chunk length in characters (~4 characters per token, so 2000 ≈ 500 tokens).</summary>
    public int ChunkSize { get; set; } = 2000;

    /// <summary>How many trailing characters of the previous chunk are repeated at the start of the next one.</summary>
    public int ChunkOverlap { get; set; } = 200;

    public void Validate()
    {
        if (ChunkSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(ChunkSize), ChunkSize, "Chunk size must be positive.");
        if (ChunkOverlap < 0)
            throw new ArgumentOutOfRangeException(nameof(ChunkOverlap), ChunkOverlap, "Chunk overlap cannot be negative.");
        if (ChunkOverlap >= ChunkSize)
            throw new ArgumentOutOfRangeException(nameof(ChunkOverlap), ChunkOverlap, "Chunk overlap must be smaller than the chunk size.");
    }
}
