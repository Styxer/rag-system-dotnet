using System.ClientModel;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;
using OpenAI;
using Rag.Core;
using Rag.Core.Chunking;
using Rag.Core.Indexing;
using Rag.Core.Retrieval;
using Rag.Core.Storage;
using Rag.Infrastructure;

namespace Rag.Api;

public sealed class OpenAISettings
{
    public const string SectionName = "OpenAI";

    public string? ApiKey { get; set; }

    /// <summary>
    /// Optional. Point at any OpenAI-compatible endpoint (Azure OpenAI v1, Ollama's /v1, LM Studio…).
    /// Leave empty for api.openai.com.
    /// </summary>
    public string? Endpoint { get; set; }

    public string ChatModel { get; set; } = "gpt-4o";
    public string EmbeddingModel { get; set; } = "text-embedding-3-small";
}

public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddRagSystem(this IServiceCollection services, IConfiguration configuration)
    {
        services.Configure<RagOptions>(configuration.GetSection(RagOptions.SectionName));
        services.Configure<OpenAISettings>(configuration.GetSection(OpenAISettings.SectionName));

        var chunking = configuration.GetSection(ChunkingOptions.SectionName).Get<ChunkingOptions>() ?? new ChunkingOptions();
        services.AddSingleton<ITextChunker>(new TextChunker(chunking));

        services.AddAiClients();
        services.AddVectorStore(configuration);

        services.AddScoped<IDocumentIndexer, DocumentIndexer>();
        services.AddScoped<IReranker, LlmReranker>();
        services.AddScoped<IRagService, RagService>();

        return services;
    }

    private static IServiceCollection AddAiClients(this IServiceCollection services)
    {
        // Clients are created lazily, so the app (and its tests) can start without an API key.
        services.AddSingleton(sp =>
        {
            var settings = sp.GetRequiredService<IOptions<OpenAISettings>>().Value;
            if (string.IsNullOrWhiteSpace(settings.ApiKey))
                throw new InvalidOperationException(
                    "OpenAI:ApiKey is not configured. Set it with 'dotnet user-secrets set OpenAI:ApiKey <key>' or the OpenAI__ApiKey environment variable.");

            var clientOptions = new OpenAIClientOptions();
            if (!string.IsNullOrWhiteSpace(settings.Endpoint))
                clientOptions.Endpoint = new Uri(settings.Endpoint);

            return new OpenAIClient(new ApiKeyCredential(settings.ApiKey), clientOptions);
        });

        services.AddChatClient(sp =>
            {
                var settings = sp.GetRequiredService<IOptions<OpenAISettings>>().Value;
                return sp.GetRequiredService<OpenAIClient>().GetChatClient(settings.ChatModel).AsIChatClient();
            })
            .UseLogging();

        services.AddEmbeddingGenerator(sp =>
            {
                var settings = sp.GetRequiredService<IOptions<OpenAISettings>>().Value;
                return sp.GetRequiredService<OpenAIClient>().GetEmbeddingClient(settings.EmbeddingModel).AsIEmbeddingGenerator();
            })
            .UseLogging();

        return services;
    }

    private static IServiceCollection AddVectorStore(this IServiceCollection services, IConfiguration configuration)
    {
        var provider = configuration["VectorStore:Provider"] ?? "Postgres";

        if (provider.Equals("InMemory", StringComparison.OrdinalIgnoreCase))
        {
            services.AddSingleton<IVectorStore, InMemoryVectorStore>();
            return services;
        }

        var connectionString = configuration.GetConnectionString("Postgres")
            ?? throw new InvalidOperationException("ConnectionStrings:Postgres is not configured.");

        var pgOptions = configuration.GetSection(PgVectorOptions.SectionName).Get<PgVectorOptions>() ?? new PgVectorOptions();

        services.AddSingleton(PgVectorDataSource.Create(connectionString));
        services.AddSingleton(pgOptions);
        services.AddSingleton<IVectorStore, PgVectorStore>();
        return services;
    }
}
