using OpenAI;
using Rag.Api;
using Rag.Api.Endpoints;
using Rag.Core.Storage;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddProblemDetails();
builder.Services.AddRagSystem(builder.Configuration);

var app = builder.Build();

app.UseExceptionHandler();
app.UseStatusCodePages();

// Fail at startup, with a clear message, when the OpenAI key is missing.
// (Tests run in the "Testing" environment with fake AI clients and need no key.)
if (!app.Environment.IsEnvironment("Testing"))
    app.Services.GetRequiredService<OpenAIClient>();

// Create the pgvector extension, table and indexes on startup.
await app.Services.GetRequiredService<IVectorStore>().InitializeAsync();

app.MapGet("/health", () => Results.Ok(new { status = "healthy" }));
app.MapRagEndpoints();

app.Run();

public partial class Program;
