using Npgsql;
using Pgvector.Npgsql;

namespace Rag.Infrastructure;

public static class PgVectorDataSource
{
    /// <summary>Creates an <see cref="NpgsqlDataSource"/> with pgvector type mapping enabled.</summary>
    public static NpgsqlDataSource Create(string connectionString)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(connectionString);
        var builder = new NpgsqlDataSourceBuilder(connectionString);
        builder.UseVector();
        return builder.Build();
    }
}
