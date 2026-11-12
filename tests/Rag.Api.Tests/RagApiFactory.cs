using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
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
    private readonly IReadOnlyDictionary<string, string?> _settings;

    public RagApiFactory(IReadOnlyDictionary<string, string?>? settings = null) =>
        _settings = settings ?? new Dictionary<string, string?>();

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
        foreach (var (key, value) in _settings)
            builder.UseSetting(key, value);

        builder.ConfigureServices(services =>
        {
            services.RemoveAll<IVectorStore>();
            services.AddSingleton<IVectorStore, InMemoryVectorStore>();

            services.RemoveAll<IChatClient>();
            services.RemoveAll<IEmbeddingGenerator<string, Embedding<float>>>();
            services.AddSingleton<IChatClient>(ChatClient);
            services.AddSingleton<IEmbeddingGenerator<string, Embedding<float>>>(EmbeddingGenerator);
        });

        // Replace JWT sign-in with a header-driven test scheme; authorization rules stay the real ones.
        builder.ConfigureTestServices(services =>
            services.AddAuthentication(options =>
            {
                options.DefaultScheme = TestAuthHandler.Scheme;
                options.DefaultAuthenticateScheme = TestAuthHandler.Scheme;
                options.DefaultChallengeScheme = TestAuthHandler.Scheme;
            }).AddScheme<AuthenticationSchemeOptions, TestAuthHandler>(TestAuthHandler.Scheme, _ => { }));
    }

    /// <summary>A client that is signed in as <paramref name="user"/> with the given groups.</summary>
    public HttpClient CreateClientAs(string user, params string[] groups)
    {
        var client = CreateClient();
        client.DefaultRequestHeaders.Add(TestAuthHandler.UserHeader, user);
        if (groups.Length > 0)
            client.DefaultRequestHeaders.Add(TestAuthHandler.GroupsHeader, string.Join(',', groups));
        return client;
    }
}
