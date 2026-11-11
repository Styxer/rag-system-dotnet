using System.Text;

namespace Rag.Core.Chunking;

public interface ITextChunker
{
    IReadOnlyList<DocumentChunk> Chunk(string sourceId, string text);
}

/// <summary>
/// Sentence-aware chunker. Text is split into sentences (on ". ", "! ", "? " and line breaks,
/// so list items, headings and table rows stay whole),
/// sentences are packed into chunks of at most <see cref="ChunkingOptions.ChunkSize"/> characters,
/// and each new chunk starts with the trailing sentences of the previous chunk that fit inside
/// <see cref="ChunkingOptions.ChunkOverlap"/> characters, so context is not lost at the boundaries.
/// A single sentence longer than the chunk size is split on character boundaries.
/// </summary>
public sealed class TextChunker : ITextChunker
{
    private readonly ChunkingOptions _options;

    public TextChunker(ChunkingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        _options = options;
    }

    public IReadOnlyList<DocumentChunk> Chunk(string sourceId, string text)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceId);

        var chunks = new List<DocumentChunk>();
        if (string.IsNullOrWhiteSpace(text))
            return chunks;

        var current = new List<string>();
        var currentLength = 0;

        void Emit(string chunkText)
        {
            chunks.Add(new DocumentChunk(DocumentChunk.CreateId(sourceId, chunks.Count), sourceId, chunks.Count, chunkText));
        }

        void Flush()
        {
            if (current.Count > 0)
                Emit(string.Join(' ', current));
        }

        foreach (var sentence in SplitSentences(text))
        {
            if (sentence.Length > _options.ChunkSize)
            {
                Flush();
                foreach (var piece in HardSplit(sentence))
                    Emit(piece);

                current.Clear();
                currentLength = 0;
                continue;
            }

            var lengthWithSentence = currentLength + (current.Count > 0 ? 1 : 0) + sentence.Length;
            if (lengthWithSentence > _options.ChunkSize && current.Count > 0)
            {
                Flush();
                current = TakeOverlap(current);
                currentLength = JoinedLength(current);

                // Drop overlap sentences until the new sentence fits.
                while (current.Count > 0 && currentLength + 1 + sentence.Length > _options.ChunkSize)
                {
                    current.RemoveAt(0);
                    currentLength = JoinedLength(current);
                }
            }

            current.Add(sentence);
            currentLength = JoinedLength(current);
        }

        Flush();
        return chunks;
    }

    /// <summary>Splits text into trimmed, non-empty sentences.</summary>
    internal static IReadOnlyList<string> SplitSentences(string text)
    {
        var sentences = new List<string>();
        var builder = new StringBuilder();
        var normalized = text.Replace("\r\n", "\n");

        void Commit()
        {
            var sentence = builder.ToString().Trim();
            if (sentence.Length > 0)
                sentences.Add(sentence);
            builder.Clear();
        }

        for (var i = 0; i < normalized.Length; i++)
        {
            var c = normalized[i];
            var next = i + 1 < normalized.Length ? normalized[i + 1] : '\0';

            if (c == '\n')
            {
                Commit();
                continue;
            }

            builder.Append(c);

            if ((c is '.' or '!' or '?') && next == ' ')
                Commit();
        }

        Commit();
        return sentences;
    }

    private IEnumerable<string> HardSplit(string sentence)
    {
        var step = _options.ChunkSize - _options.ChunkOverlap;
        for (var start = 0; start < sentence.Length; start += step)
        {
            var length = Math.Min(_options.ChunkSize, sentence.Length - start);
            yield return sentence.Substring(start, length);
            if (start + length >= sentence.Length)
                yield break;
        }
    }

    private List<string> TakeOverlap(List<string> sentences)
    {
        var overlap = new List<string>();
        if (_options.ChunkOverlap == 0)
            return overlap;

        var length = 0;
        for (var i = sentences.Count - 1; i >= 0; i--)
        {
            var added = sentences[i].Length + (overlap.Count > 0 ? 1 : 0);
            if (length + added > _options.ChunkOverlap)
                break;
            overlap.Insert(0, sentences[i]);
            length += added;
        }

        return overlap;
    }

    private static int JoinedLength(List<string> sentences) =>
        sentences.Count == 0 ? 0 : sentences.Sum(s => s.Length) + sentences.Count - 1;
}
