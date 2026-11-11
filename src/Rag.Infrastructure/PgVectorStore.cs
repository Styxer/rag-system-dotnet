using System.Text.Json;
using Npgsql;
using Pgvector;
using Rag.Core.Storage;

namespace Rag.Infrastructure;

public sealed class PgVectorOptions
{
    public const string SectionName = "PgVector";

    /// <summary>Must match the embedding model (text-embedding-3-small = 1536).</summary>
    public int Dimensions { get; set; } = 1536;

    public string TableName { get; set; } = "document_embeddings";

    /// <summary>Text-search configuration used for hybrid search (e.g. 'english', 'simple').</summary>
    public string TextSearchConfig { get; set; } = "english";
}

/// <summary>
/// PostgreSQL + pgvector store. Uses an HNSW index with cosine distance (HNSW can be built on an
/// empty table, unlike IVFFlat) and a generated tsvector column + GIN index for hybrid search.
/// The <see cref="NpgsqlDataSource"/> must be built with <c>UseVector()</c>.
/// </summary>
public sealed class PgVectorStore : IVectorStore
{
    /// <summary>Each side of a hybrid search fetches topK × this many candidates before fusion.</summary>
    internal const int HybridCandidateMultiplier = 4;

    // pgvector's limits for hnsw.ef_search (how many candidates an HNSW scan can return).
    private const int DefaultEfSearch = 40;
    private const int MaxEfSearch = 1000;

    private readonly NpgsqlDataSource _dataSource;
    private readonly PgVectorOptions _options;
    private readonly string _table;
    private readonly string _upsertSql;

    public PgVectorStore(NpgsqlDataSource dataSource, PgVectorOptions options)
    {
        _dataSource = dataSource;
        _options = options;
        _table = QuoteIdentifier(options.TableName);

        if (options.TextSearchConfig.Length == 0 || !options.TextSearchConfig.All(char.IsAsciiLetterLower))
            throw new ArgumentException($"Invalid text search config '{options.TextSearchConfig}'.");

        _upsertSql = $"""
            INSERT INTO {_table} (id, source, chunk_index, text, embedding, metadata)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                source = EXCLUDED.source,
                chunk_index = EXCLUDED.chunk_index,
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata;
            """;
    }

    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        var sql = $"""
            CREATE EXTENSION IF NOT EXISTS vector;

            CREATE TABLE IF NOT EXISTS {_table} (
                id          TEXT PRIMARY KEY,
                source      TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text        TEXT NOT NULL,
                embedding   vector({_options.Dimensions}) NOT NULL,
                metadata    JSONB NOT NULL DEFAULT '{"{}"}'::jsonb,
                search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('{_options.TextSearchConfig}', text)) STORED,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );

            CREATE INDEX IF NOT EXISTS {IndexName("embedding_hnsw")}
                ON {_table} USING hnsw (embedding vector_cosine_ops);

            CREATE INDEX IF NOT EXISTS {IndexName("search_vector_gin")}
                ON {_table} USING gin (search_vector);

            CREATE INDEX IF NOT EXISTS {IndexName("source")}
                ON {_table} (source);
            """;

        await using (var cmd = _dataSource.CreateCommand(sql))
            await cmd.ExecuteNonQueryAsync(cancellationToken);

        // The data source caches type info; make sure it knows about the "vector" type
        // in case the extension was just created.
        await using (var conn = await _dataSource.OpenConnectionAsync(cancellationToken))
            await conn.ReloadTypesAsync();

        await EnsureDimensionsMatchAsync(cancellationToken);
    }

    /// <summary>
    /// CREATE TABLE IF NOT EXISTS keeps an existing table as it is, so a changed embedding size
    /// would only surface later as a failed insert. Fail at startup with a clear message instead.
    /// </summary>
    private async Task EnsureDimensionsMatchAsync(CancellationToken cancellationToken)
    {
        // For a pgvector column, atttypmod is the declared number of dimensions.
        const string sql = """
            SELECT atttypmod FROM pg_attribute
            WHERE attrelid = to_regclass($1) AND attname = 'embedding' AND NOT attisdropped;
            """;

        await using var cmd = _dataSource.CreateCommand(sql);
        cmd.Parameters.Add(new NpgsqlParameter { Value = _table });
        var actual = await cmd.ExecuteScalarAsync(cancellationToken);

        if (actual is int dimensions && dimensions != _options.Dimensions)
            throw new InvalidOperationException(
                $"Table {_options.TableName} stores {dimensions}-dimensional embeddings, but PgVector:Dimensions is {_options.Dimensions}. " +
                "Use the embedding model the table was created for, or point PgVector:TableName at a new table and re-index.");
    }

    public async Task UpsertAsync(IReadOnlyCollection<VectorRecord> records, CancellationToken cancellationToken = default)
    {
        if (records.Count == 0)
            return;

        await using var batch = _dataSource.CreateBatch();
        AddUpsertCommands(batch, records);
        await batch.ExecuteNonQueryAsync(cancellationToken);
    }

    public async Task ReplaceSourceAsync(string source, IReadOnlyCollection<VectorRecord> records, CancellationToken cancellationToken = default)
    {
        await using var conn = await _dataSource.OpenConnectionAsync(cancellationToken);
        await using var tx = await conn.BeginTransactionAsync(cancellationToken);

        // Delete + insert in one transaction: if anything fails, the old chunks stay.
        await using var batch = new NpgsqlBatch(conn, tx);
        var delete = new NpgsqlBatchCommand($"DELETE FROM {_table} WHERE source = $1");
        delete.Parameters.Add(new NpgsqlParameter { Value = source });
        batch.BatchCommands.Add(delete);
        AddUpsertCommands(batch, records);

        await batch.ExecuteNonQueryAsync(cancellationToken);
        await tx.CommitAsync(cancellationToken);
    }

    public async Task<int> DeleteBySourceAsync(string source, CancellationToken cancellationToken = default)
    {
        await using var cmd = _dataSource.CreateCommand($"DELETE FROM {_table} WHERE source = $1");
        cmd.Parameters.Add(new NpgsqlParameter { Value = source });
        return await cmd.ExecuteNonQueryAsync(cancellationToken);
    }

    public Task<IReadOnlyList<SearchResult>> SearchAsync(
        ReadOnlyMemory<float> queryEmbedding,
        int topK,
        CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(topK);

        // <=> is cosine distance; similarity = 1 - distance.
        var sql = $"""
            SELECT id, source, chunk_index, text, 1 - (embedding <=> $1) AS score
            FROM {_table}
            ORDER BY embedding <=> $1
            LIMIT $2;
            """;

        return QueryAsync(
            sql,
            candidateLimit: topK,
            [new Vector(queryEmbedding), topK],
            cancellationToken);
    }

    public Task<IReadOnlyList<SearchResult>> HybridSearchAsync(
        ReadOnlyMemory<float> queryEmbedding,
        string queryText,
        int topK,
        HybridSearchWeights weights,
        CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(topK);
        ArgumentNullException.ThrowIfNull(weights);

        // Weighted Reciprocal Rank Fusion: each side ranks its own candidates, and a chunk scores
        // weight / (60 + rank) per list it appears in. Only ranks are combined, never raw scores,
        // because cosine similarity (0..1) and ts_rank (usually < 0.1) are not comparable.
        var sql = $"""
            WITH vector_results AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY distance, id) AS rank
                FROM (
                    SELECT id, embedding <=> $1 AS distance
                    FROM {_table}
                    ORDER BY distance
                    LIMIT $3
                ) nearest
            ),
            keyword_results AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY keyword_rank DESC, id) AS rank
                FROM (
                    SELECT id, ts_rank(search_vector, plainto_tsquery('{_options.TextSearchConfig}', $2)) AS keyword_rank
                    FROM {_table}
                    WHERE search_vector @@ plainto_tsquery('{_options.TextSearchConfig}', $2)
                    ORDER BY keyword_rank DESC
                    LIMIT $3
                ) matched
            )
            SELECT d.id, d.source, d.chunk_index, d.text,
                   COALESCE($4 / ({HybridSearchWeights.RrfK} + v.rank), 0)
                 + COALESCE($5 / ({HybridSearchWeights.RrfK} + k.rank), 0) AS score
            FROM vector_results v
            FULL OUTER JOIN keyword_results k ON v.id = k.id
            JOIN {_table} d ON d.id = COALESCE(v.id, k.id)
            ORDER BY score DESC, d.id
            LIMIT $6;
            """;

        var pool = topK * HybridCandidateMultiplier;
        var scale = weights.Scale;

        return QueryAsync(
            sql,
            candidateLimit: pool,
            [new Vector(queryEmbedding), queryText, pool, weights.VectorWeight * scale, weights.KeywordWeight * scale, topK],
            cancellationToken);
    }

    /// <summary>
    /// Runs a search inside a short transaction that first raises <c>hnsw.ef_search</c>.
    /// An HNSW index scan returns at most ef_search rows (40 by default), so without this a
    /// request for more candidates than that would silently come back short.
    /// </summary>
    private async Task<IReadOnlyList<SearchResult>> QueryAsync(
        string sql,
        int candidateLimit,
        object[] parameters,
        CancellationToken cancellationToken)
    {
        await using var conn = await _dataSource.OpenConnectionAsync(cancellationToken);
        await using var tx = await conn.BeginTransactionAsync(cancellationToken);

        // SET does not accept bind parameters; the value is a clamped integer, so this is safe.
        var efSearch = Math.Clamp(candidateLimit, DefaultEfSearch, MaxEfSearch);
        await using (var set = new NpgsqlCommand($"SET LOCAL hnsw.ef_search = {efSearch}", conn, tx))
            await set.ExecuteNonQueryAsync(cancellationToken);

        var results = new List<SearchResult>();
        await using (var cmd = new NpgsqlCommand(sql, conn, tx))
        {
            foreach (var value in parameters)
                cmd.Parameters.Add(new NpgsqlParameter { Value = value });

            await using var reader = await cmd.ExecuteReaderAsync(cancellationToken);
            while (await reader.ReadAsync(cancellationToken))
            {
                results.Add(new SearchResult(
                    reader.GetString(0),
                    reader.GetString(1),
                    reader.GetInt32(2),
                    reader.GetString(3),
                    Convert.ToDouble(reader.GetValue(4))));
            }
        }

        await tx.CommitAsync(cancellationToken);
        return results;
    }

    private void AddUpsertCommands(NpgsqlBatch batch, IReadOnlyCollection<VectorRecord> records)
    {
        foreach (var record in records)
        {
            var command = new NpgsqlBatchCommand(_upsertSql);
            command.Parameters.Add(new NpgsqlParameter { Value = record.Id });
            command.Parameters.Add(new NpgsqlParameter { Value = record.Source });
            command.Parameters.Add(new NpgsqlParameter { Value = record.ChunkIndex });
            command.Parameters.Add(new NpgsqlParameter { Value = record.Text });
            command.Parameters.Add(new NpgsqlParameter { Value = new Vector(record.Embedding) });
            command.Parameters.Add(new NpgsqlParameter
            {
                Value = JsonSerializer.Serialize(new { source = record.Source, chunkIndex = record.ChunkIndex })
            });
            batch.BatchCommands.Add(command);
        }
    }

    private string IndexName(string suffix) => QuoteIdentifier($"ix_{_options.TableName}_{suffix}");

    internal static string QuoteIdentifier(string identifier)
    {
        if (string.IsNullOrWhiteSpace(identifier) || !identifier.All(c => char.IsAsciiLetterOrDigit(c) || c == '_'))
            throw new ArgumentException($"Invalid SQL identifier '{identifier}'. Use letters, digits and underscores only.");
        return $"\"{identifier}\"";
    }
}
