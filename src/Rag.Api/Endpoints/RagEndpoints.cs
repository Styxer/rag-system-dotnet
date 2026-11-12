using System.Security.Claims;
using Microsoft.Extensions.Options;
using Rag.Api.Security;
using Rag.Core;
using Rag.Core.Indexing;
using Rag.Core.Security;
using Rag.Core.Storage;

namespace Rag.Api.Endpoints;

/// <param name="AllowedPrincipals">
/// Extra principals (<c>user:{id}</c> or <c>group:{name}</c>) that may read the document.
/// The uploader always can; by default nobody else can.
/// </param>
public sealed record IndexDocumentRequest(string FileName, string Content, IReadOnlyList<string>? AllowedPrincipals = null);

public sealed record IndexDocumentResponse(string Message, string DocumentId, int ChunkCount);

public sealed record QueryRequest(string Question, int? TopK = null, bool? UseHybridSearch = null, bool? UseReranking = null);

public static class RagEndpoints
{
    public const int MaxTopK = 50;
    public const int MaxFileNameLength = 200;

    /// <summary>Upper bound for one document (characters). Every upload costs embedding calls.</summary>
    public const int MaxContentLength = 1_000_000;

    public static IEndpointRouteBuilder MapRagEndpoints(this IEndpointRouteBuilder app)
    {
        var api = app.MapGroup("/api");

        api.MapPost("/documents", async (
            IndexDocumentRequest request,
            ClaimsPrincipal user,
            IOptions<AuthSettings> auth,
            IDocumentIndexer indexer,
            CancellationToken ct) =>
        {
            if (!user.TryGetAccessScope(auth.Value.GroupClaim, out var scope))
                return NoUserId();

            var errors = new Dictionary<string, string[]>();
            if (string.IsNullOrWhiteSpace(request.FileName))
                errors[nameof(request.FileName)] = ["File name is required."];
            else if (request.FileName.Trim().Length > MaxFileNameLength || request.FileName.Any(char.IsControl))
                errors[nameof(request.FileName)] = [$"File name must be at most {MaxFileNameLength} characters with no control characters."];

            if (string.IsNullOrWhiteSpace(request.Content))
                errors[nameof(request.Content)] = ["Content is required."];
            else if (request.Content.Length > MaxContentLength)
                errors[nameof(request.Content)] = [$"Content must be at most {MaxContentLength:N0} characters."];

            if (!scope.TryBuildAcl(request.AllowedPrincipals, out _, out var aclError))
                errors[nameof(request.AllowedPrincipals)] = [aclError!];

            if (errors.Count > 0)
                return Results.ValidationProblem(errors);

            try
            {
                var result = await indexer.IndexDocumentAsync(
                    request.FileName.Trim(), request.Content, scope, request.AllowedPrincipals?.ToList(), ct);
                return Results.Ok(new IndexDocumentResponse("Document indexed successfully", result.DocumentId, result.ChunkCount));
            }
            catch (SourceOwnershipException)
            {
                return Results.Problem(
                    statusCode: StatusCodes.Status409Conflict,
                    title: "Document id already in use",
                    detail: "A document with this file name belongs to another owner. Choose a different file name.");
            }
        })
        .WithName("IndexDocument")
        .RequireRateLimiting(RateLimitPolicies.Upload);

        api.MapPost("/query", async (
            QueryRequest request,
            ClaimsPrincipal user,
            IOptions<AuthSettings> auth,
            IRagService rag,
            CancellationToken ct) =>
        {
            if (!user.TryGetAccessScope(auth.Value.GroupClaim, out var scope))
                return NoUserId();

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
                scope,
                new QueryOptions(request.TopK, request.UseHybridSearch, request.UseReranking),
                ct);

            return Results.Ok(response);
        })
        .WithName("Query")
        .RequireRateLimiting(RateLimitPolicies.Query);

        return app;
    }

    private static IResult NoUserId() => Results.Problem(
        statusCode: StatusCodes.Status403Forbidden,
        title: "No user id",
        detail: "The token has no usable 'sub' claim.");
}
