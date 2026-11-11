namespace Rag.Core.Chunking;

/// <summary>
/// A piece of a source document that is embedded and stored as one vector.
/// </summary>
/// <param name="Id">Stable id in the form <c>{sourceId}-chunk-{index}</c>.</param>
/// <param name="SourceId">Id of the document this chunk came from (usually the file name).</param>
/// <param name="Index">Zero-based position of the chunk inside the document.</param>
/// <param name="Text">The chunk text.</param>
public sealed record DocumentChunk(string Id, string SourceId, int Index, string Text)
{
    public static string CreateId(string sourceId, int index) => $"{sourceId}-chunk-{index}";
}
