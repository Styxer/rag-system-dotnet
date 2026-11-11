using System.Globalization;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Rag.Core.Storage;

namespace Rag.Core.Retrieval;

public interface IReranker
{
    Task<IReadOnlyList<SearchResult>> RerankAsync(
        string question,
        IReadOnlyList<SearchResult> candidates,
        int topK,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Asks the chat model to score each candidate's relevance from 0.0 to 1.0 and reorders by that score.
/// Only the first <see cref="PreviewLength"/> characters of each candidate are sent to keep the call cheap.
/// If the model returns something unusable, the original order is kept.
/// </summary>
public sealed class LlmReranker : IReranker
{
    internal const int PreviewLength = 200;

    private readonly IChatClient _chatClient;
    private readonly ILogger<LlmReranker> _logger;

    public LlmReranker(IChatClient chatClient, ILogger<LlmReranker>? logger = null)
    {
        _chatClient = chatClient;
        _logger = logger ?? NullLogger<LlmReranker>.Instance;
    }

    public async Task<IReadOnlyList<SearchResult>> RerankAsync(
        string question,
        IReadOnlyList<SearchResult> candidates,
        int topK,
        CancellationToken cancellationToken = default)
    {
        if (candidates.Count <= 1)
            return candidates.Take(topK).ToList();

        var prompt = BuildPrompt(question, candidates);
        var response = await _chatClient.GetResponseAsync(
            [new ChatMessage(ChatRole.User, prompt)],
            new ChatOptions { Temperature = 0 },
            cancellationToken);

        var scores = ParseScores(response.Text, candidates.Count);
        if (scores is null)
        {
            _logger.LogWarning("Re-ranker returned an unusable response; keeping the original order");
            return candidates.Take(topK).ToList();
        }

        return candidates
            .Select((c, i) => c with { Score = scores[i] })
            .OrderByDescending(c => c.Score)
            .Take(topK)
            .ToList();
    }

    internal static string BuildPrompt(string question, IReadOnlyList<SearchResult> candidates)
    {
        var sb = new StringBuilder();
        sb.AppendLine("Rate how relevant each document is to the question on a scale from 0.0 (irrelevant) to 1.0 (highly relevant).");
        sb.AppendLine($"Return ONLY a JSON array of {candidates.Count} numbers, in the same order as the documents. No other text.");
        sb.AppendLine();
        sb.AppendLine($"Question: {question}");
        sb.AppendLine();
        for (var i = 0; i < candidates.Count; i++)
        {
            var text = candidates[i].Text;
            var preview = text.Length > PreviewLength ? text[..PreviewLength] : text;
            sb.AppendLine($"Document {i + 1}: {preview}");
        }
        return sb.ToString();
    }

    /// <summary>
    /// Extracts a JSON array of numbers from the model output (tolerating markdown code fences
    /// and surrounding text). Returns null if the array is missing or has the wrong length.
    /// </summary>
    internal static double[]? ParseScores(string? text, int expectedCount)
    {
        if (string.IsNullOrWhiteSpace(text))
            return null;

        var start = text.IndexOf('[');
        var end = text.LastIndexOf(']');
        if (start < 0 || end <= start)
            return null;

        try
        {
            using var doc = JsonDocument.Parse(text[start..(end + 1)]);
            if (doc.RootElement.ValueKind != JsonValueKind.Array)
                return null;

            var scores = new List<double>();
            foreach (var element in doc.RootElement.EnumerateArray())
            {
                double value;
                if (element.ValueKind == JsonValueKind.Number)
                    value = element.GetDouble();
                else if (element.ValueKind == JsonValueKind.String &&
                         double.TryParse(element.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed))
                    value = parsed;
                else
                    return null;

                scores.Add(Math.Clamp(value, 0.0, 1.0));
            }

            return scores.Count == expectedCount ? scores.ToArray() : null;
        }
        catch (JsonException)
        {
            return null;
        }
    }
}
