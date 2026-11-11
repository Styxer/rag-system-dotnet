using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Rag.Core.Storage;
using Rag.Testing;

namespace Rag.Api.Tests;

/// <summary>
/// Hosts the real API in memory with the in-memory vector store and fake AI clients,
/// so endpoint tests need neither Postgres nor an OpenAI key.
/// </summary>
public sealed class RagApiFactory : WebApplicationFactory<Program>
{
    /// <summary>Scores the fake model gives when asked to re-rank two candidates.</summary>
    public const string RerankReply = "[0.1, 0.9]";

    public FakeChatClient ChatClient { get; } = new(messages =>
    {
        if (messages[0].Text.StartsWith("Rate how relevant", StringComparison.Ordinal))
            return RerankReply;

        return messages.Any(m => m.Text.Contains("vacation days", StringComparison.OrdinalIgnoreCase))
            ? "Employees receive twenty five vacation days per year."
            : "I don't have information about that in my knowledge base.";
    });

    public FakeEmbeddingGenerator EmbeddingGenerator { get; } = new();

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.UseEnvironment("Testing");
        builder.UseSetting("VectorStore:Provider", "InMemory");

        builder.ConfigureServices(services =>
        {
            services.RemoveAll<IVectorStore>();
            services.AddSingleton<IVectorStore, InMemoryVectorStore>();

            services.RemoveAll<IChatClient>();
            services.RemoveAll<IEmbeddingGenerator<string, Embedding<float>>>();
            services.AddSingleton<IChatClient>(ChatClient);
            services.AddSingleton<IEmbeddingGenerator<string, Embedding<float>>>(EmbeddingGenerator);
        });
    }
}
