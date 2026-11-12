using System.Net;
using System.Net.Http.Json;
using Rag.Api.Endpoints;
using Rag.Core;
using Rag.Core.Generation;
using Rag.Testing;

namespace Rag.Api.Tests;

public class SecurityTests
{
    private const string Question = "How many vacation days do employees receive?";

    private static Task<HttpResponseMessage> Upload(
        HttpClient client, string fileName = SampleDocuments.HandbookName, string[]? shareWith = null) =>
        client.PostAsJsonAsync("/api/documents",
            new IndexDocumentRequest(fileName, SampleDocuments.Handbook, shareWith));

    private static async Task<RagResponse> Ask(HttpClient client)
    {
        var response = await client.PostAsJsonAsync("/api/query", new QueryRequest(Question, TopK: 3));
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        return (await response.Content.ReadFromJsonAsync<RagResponse>())!;
    }

    [Fact]
    public async Task Health_IsAnonymous()
    {
        await using var factory = new RagApiFactory();

        var response = await factory.CreateClient().GetAsync("/health");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task Endpoints_RequireSignIn()
    {
        await using var factory = new RagApiFactory();
        var anonymous = factory.CreateClient();

        var upload = await Upload(anonymous);
        var query = await anonymous.PostAsJsonAsync("/api/query", new QueryRequest(Question));

        Assert.Equal(HttpStatusCode.Unauthorized, upload.StatusCode);
        Assert.Equal(HttpStatusCode.Unauthorized, query.StatusCode);
        Assert.Equal(0, factory.EmbeddingGenerator.CallCount);
    }

    [Fact]
    public async Task UploadedDocument_IsPrivateByDefault()
    {
        await using var factory = new RagApiFactory();
        await Upload(factory.CreateClientAs("alice"));

        var forAlice = await Ask(factory.CreateClientAs("alice"));
        var forBob = await Ask(factory.CreateClientAs("bob"));

        Assert.NotEmpty(forAlice.Sources);
        Assert.Empty(forBob.Sources);
        Assert.Equal(PromptBuilder.NoInformationAnswer, forBob.Answer);
    }

    [Fact]
    public async Task UploadedDocument_CanBeSharedWithAGroup()
    {
        await using var factory = new RagApiFactory();
        await Upload(factory.CreateClientAs("alice"), shareWith: ["group:hr"]);

        var hrMember = await Ask(factory.CreateClientAs("carol", "hr", "staff"));
        var outsider = await Ask(factory.CreateClientAs("dave", "staff"));

        Assert.NotEmpty(hrMember.Sources);
        Assert.Empty(outsider.Sources);
    }

    [Fact]
    public async Task UploadedDocument_CanBeSharedWithAUser()
    {
        await using var factory = new RagApiFactory();
        await Upload(factory.CreateClientAs("alice"), shareWith: ["user:bob"]);

        Assert.NotEmpty((await Ask(factory.CreateClientAs("bob"))).Sources);
        Assert.Empty((await Ask(factory.CreateClientAs("dave"))).Sources);
    }

    [Fact]
    public async Task Overwriting_ADocumentOfAnotherUser_ReturnsConflict_AndKeepsIt()
    {
        await using var factory = new RagApiFactory();
        await Upload(factory.CreateClientAs("alice"));

        var response = await factory.CreateClientAs("bob").PostAsJsonAsync("/api/documents",
            new IndexDocumentRequest(SampleDocuments.HandbookName, "Bob's replacement text."));

        Assert.Equal(HttpStatusCode.Conflict, response.StatusCode);
        Assert.NotEmpty((await Ask(factory.CreateClientAs("alice"))).Sources);
    }

    [Fact]
    public async Task Owner_CanReindexAndChangeSharing()
    {
        await using var factory = new RagApiFactory();
        var alice = factory.CreateClientAs("alice");
        await Upload(alice);

        var response = await Upload(alice, shareWith: ["user:bob"]);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.NotEmpty((await Ask(factory.CreateClientAs("bob"))).Sources);
    }

    [Theory]
    [InlineData("everyone")]
    [InlineData("user:")]
    [InlineData("role:admin")]
    public async Task Upload_InvalidPrincipal_ReturnsBadRequest(string principal)
    {
        await using var factory = new RagApiFactory();

        var response = await Upload(factory.CreateClientAs("alice"), shareWith: [principal]);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal(0, factory.EmbeddingGenerator.CallCount);
    }

    [Fact]
    public async Task Upload_TooManyPrincipals_ReturnsBadRequest()
    {
        await using var factory = new RagApiFactory();
        var many = Enumerable.Range(0, 60).Select(i => $"user:u{i}").ToArray();

        var response = await Upload(factory.CreateClientAs("alice"), shareWith: many);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Theory]
    [InlineData(201)]
    public async Task Upload_FileNameTooLong_ReturnsBadRequest(int length)
    {
        await using var factory = new RagApiFactory();

        var response = await Upload(factory.CreateClientAs("alice"), new string('a', length));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task Upload_FileNameWithControlCharacters_ReturnsBadRequest()
    {
        await using var factory = new RagApiFactory();

        var response = await Upload(factory.CreateClientAs("alice"), "bad\nname.txt");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task Queries_AreRateLimitedPerUser()
    {
        await using var factory = new RagApiFactory(new Dictionary<string, string?> { ["RateLimiting:QueryPerMinute"] = "2" });
        var alice = factory.CreateClientAs("alice");

        for (var i = 0; i < 2; i++)
            Assert.Equal(HttpStatusCode.OK, (await alice.PostAsJsonAsync("/api/query", new QueryRequest(Question))).StatusCode);

        var limited = await alice.PostAsJsonAsync("/api/query", new QueryRequest(Question));
        var otherUser = await factory.CreateClientAs("bob").PostAsJsonAsync("/api/query", new QueryRequest(Question));

        Assert.Equal(HttpStatusCode.TooManyRequests, limited.StatusCode);
        Assert.True(limited.Headers.Contains("Retry-After"));
        Assert.Equal(HttpStatusCode.OK, otherUser.StatusCode);
    }

    [Fact]
    public async Task Uploads_AreRateLimitedSeparatelyFromQueries()
    {
        await using var factory = new RagApiFactory(new Dictionary<string, string?>
        {
            ["RateLimiting:UploadPerMinute"] = "1",
            ["RateLimiting:QueryPerMinute"] = "5",
        });
        var alice = factory.CreateClientAs("alice");

        Assert.Equal(HttpStatusCode.OK, (await Upload(alice)).StatusCode);
        Assert.Equal(HttpStatusCode.TooManyRequests, (await Upload(alice)).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await alice.PostAsJsonAsync("/api/query", new QueryRequest(Question))).StatusCode);
    }
}
