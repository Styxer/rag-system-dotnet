using System.Net;
using System.Net.Http.Json;
using Rag.Api.Endpoints;
using Rag.Core;
using Rag.Core.Generation;
using Rag.Testing;

namespace Rag.Api.Tests;

public class RagEndpointsTests
{
    [Fact]
    public async Task Health_ReturnsOk()
    {
        await using var factory = new RagApiFactory();
        var client = factory.CreateClient();

        var response = await client.GetAsync("/health");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task IndexDocument_ValidRequest_ReturnsChunkCount()
    {
        await using var factory = new RagApiFactory();
        var client = factory.CreateClient();

        var response = await client.PostAsJsonAsync("/api/documents",
            new IndexDocumentRequest(SampleDocuments.HandbookName, SampleDocuments.Handbook));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var body = await response.Content.ReadFromJsonAsync<IndexDocumentResponse>();
        Assert.NotNull(body);
        Assert.Equal("Document indexed successfully", body.Message);
        Assert.Equal(SampleDocuments.HandbookName, body.DocumentId);
        Assert.True(body.ChunkCount >= 1);
    }

    [Theory]
    [InlineData("", "content")]
    [InlineData("file.md", "")]
    [InlineData("  ", "  ")]
    public async Task IndexDocument_MissingFields_ReturnsBadRequest(string fileName, string content)
    {
        await using var factory = new RagApiFactory();
        var client = factory.CreateClient();

        var response = await client.PostAsJsonAsync("/api/documents", new IndexDocumentRequest(fileName, content));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task Query_AfterIndexing_ReturnsGroundedAnswerWithSources()
    {
        await using var factory = new RagApiFactory();
        var client = factory.CreateClient();

        await client.PostAsJsonAsync("/api/documents", new IndexDocumentRequest(SampleDocuments.HandbookName, SampleDocuments.Handbook));
        await client.PostAsJsonAsync("/api/documents", new IndexDocumentRequest(SampleDocuments.RecipesName, SampleDocuments.Recipes));

        var response = await client.PostAsJsonAsync("/api/query",
            new QueryRequest("How many vacation days do employees receive?", TopK: 2));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var body = await response.Content.ReadFromJsonAsync<RagResponse>();
        Assert.NotNull(body);
        Assert.Equal("Employees receive twenty five vacation days per year.", body.Answer);
        Assert.InRange(body.Sources.Count, 1, 2);
        Assert.Equal(SampleDocuments.HandbookName, body.Sources[0].DocumentName);
    }

    [Fact]
    public async Task Query_WithReranking_ReordersSourcesByModelScores()
    {
        await using var factory = new RagApiFactory();
        var client = factory.CreateClient();

        await client.PostAsJsonAsync("/api/documents", new IndexDocumentRequest(SampleDocuments.HandbookName, SampleDocuments.Handbook));
        await client.PostAsJsonAsync("/api/documents", new IndexDocumentRequest(SampleDocuments.RecipesName, SampleDocuments.Recipes));
        const string question = "How many vacation days do employees receive?";

        var plain = await (await client.PostAsJsonAsync("/api/query",
            new QueryRequest(question, TopK: 2, UseHybridSearch: true))).Content.ReadFromJsonAsync<RagResponse>();
        var reranked = await (await client.PostAsJsonAsync("/api/query",
            new QueryRequest(question, TopK: 2, UseHybridSearch: true, UseReranking: true))).Content.ReadFromJsonAsync<RagResponse>();

        // Retrieval puts the handbook first; the fake model scores the second candidate higher (0.1 vs 0.9).
        Assert.NotNull(plain);
        Assert.NotNull(reranked);
        Assert.Equal(new[] { SampleDocuments.HandbookName, SampleDocuments.RecipesName }, plain.Sources.Select(s => s.DocumentName));
        Assert.Equal(new[] { SampleDocuments.RecipesName, SampleDocuments.HandbookName }, reranked.Sources.Select(s => s.DocumentName));
        Assert.Equal(new[] { 0.9, 0.1 }, reranked.Sources.Select(s => s.Relevance));
    }

    [Fact]
    public async Task IndexDocument_ContentOverLimit_ReturnsBadRequest()
    {
        await using var factory = new RagApiFactory();
        var client = factory.CreateClient();

        var response = await client.PostAsJsonAsync("/api/documents",
            new IndexDocumentRequest("big.txt", new string('a', RagEndpoints.MaxContentLength + 1)));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal(0, factory.EmbeddingGenerator.CallCount);
    }

    [Fact]
    public async Task Query_EmptyKnowledgeBase_ReturnsNoInformationAnswer()
    {
        await using var factory = new RagApiFactory();
        var client = factory.CreateClient();

        var response = await client.PostAsJsonAsync("/api/query", new QueryRequest("What is the meaning of life?"));

        var body = await response.Content.ReadFromJsonAsync<RagResponse>();
        Assert.NotNull(body);
        Assert.Equal(PromptBuilder.NoInformationAnswer, body.Answer);
        Assert.Empty(body.Sources);
        Assert.Equal(0, factory.ChatClient.CallCount);
    }

    [Theory]
    [InlineData("", null)]
    [InlineData("valid question", 0)]
    [InlineData("valid question", 51)]
    public async Task Query_InvalidRequest_ReturnsBadRequest(string question, int? topK)
    {
        await using var factory = new RagApiFactory();
        var client = factory.CreateClient();

        var response = await client.PostAsJsonAsync("/api/query", new QueryRequest(question, topK));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }
}
