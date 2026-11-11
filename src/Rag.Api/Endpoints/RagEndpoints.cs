using Rag.Core;
using Rag.Core.Indexing;

namespace Rag.Api.Endpoints;

public sealed record IndexDocumentRequest(string FileName, string Content);

public sealed record IndexDocumentResponse(string Message, string DocumentId, int ChunkCount);

public sealed record QueryRequest(string Question, int? TopK = null, bool? UseHybridSearch = null, bool? UseReranking = null);

public static class RagEndpoints
{
    public const int MaxTopK = 50;

    /// <summary>Upper bound for one document (characters). Every upload costs embedding calls.</summary>
    public const int MaxContentLength = 1_000_000;

    public static IEndpointRouteBuilder MapRagEndpoints(this IEndpointRouteBuilder app)
    {
        var api = app.MapGroup("/api");

        api.MapPost("/documents", async (IndexDocumentRequest request, IDocumentIndexer indexer, CancellationToken ct) =>
        {
            var errors = new Dictionary<string, string[]>();
            if (string.IsNullOrWhiteSpace(request.FileName))
                errors[nameof(request.FileName)] = ["File name is required."];
            if (string.IsNullOrWhiteSpace(request.Content))
                errors[nameof(request.Content)] = ["Content is required."];
            else if (request.Content.Length > MaxContentLength)
                errors[nameof(request.Content)] = [$"Content must be at most {MaxContentLength:N0} characters."];
            if (errors.Count > 0)
                return Results.ValidationProblem(errors);

            var result = await indexer.IndexDocumentAsync(request.FileName.Trim(), request.Content, ct);
            return Results.Ok(new IndexDocumentResponse("Document indexed successfully", result.DocumentId, result.ChunkCount));
        })
        .WithName("IndexDocument");

        api.MapPost("/query", async (QueryRequest request, IRagService rag, CancellationToken ct) =>
        {
            if (string.IsNullOrWhiteSpace(request.Question))
                return Results.ValidationProblem(new Dictionary<string, string[]>
                {
                    [nameof(request.Question)] = ["Question is required."],
                });

            if (request.TopK is < 1 or > MaxTopK)
                return Results.ValidationProblem(new Dictionary<string, string[]>
                {
                    [nameof(request.TopK)] = [$"TopK must be between 1 and {MaxTopK}."],
                });

            var response = await rag.AskAsync(
                request.Question,
                new QueryOptions(request.TopK, request.UseHybridSearch, request.UseReranking),
                ct);

            return Results.Ok(response);
        })
        .WithName("Query");

        return app;
    }
}
