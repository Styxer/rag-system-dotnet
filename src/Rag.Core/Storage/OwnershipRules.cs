using Rag.Core.Security;

namespace Rag.Core.Storage;

/// <summary>Thrown when a caller tries to replace or delete a document that another owner uploaded.</summary>
public sealed class SourceOwnershipException(string source)
    : Exception($"Document '{source}' belongs to another owner.")
{
    public string Source { get; } = source;
}

/// <summary>Write rules shared by every <see cref="IVectorStore"/> implementation.</summary>
public static class OwnershipRules
{
    /// <summary>
    /// A document id belongs to whoever uploaded it first. Only that owner may replace it, and a caller
    /// may only write records that name the document and themselves. The unrestricted scope skips the checks.
    /// </summary>
    public static void EnsureCanReplace(
        AccessScope scope,
        string source,
        string? existingOwner,
        IEnumerable<VectorRecord> newRecords)
    {
        ArgumentNullException.ThrowIfNull(scope);
        if (scope.IsUnrestricted)
            return;

        EnsureOwner(scope, source, existingOwner);

        foreach (var record in newRecords)
        {
            if (!string.Equals(record.Source, source, StringComparison.Ordinal) ||
                !string.Equals(record.Owner, scope.OwnerPrincipal, StringComparison.Ordinal))
            {
                throw new ArgumentException(
                    $"Record '{record.Id}' does not belong to document '{source}' and the calling user.",
                    nameof(newRecords));
            }
        }
    }

    public static void EnsureCanDelete(AccessScope scope, string source, string? existingOwner)
    {
        ArgumentNullException.ThrowIfNull(scope);
        if (!scope.IsUnrestricted)
            EnsureOwner(scope, source, existingOwner);
    }

    private static void EnsureOwner(AccessScope scope, string source, string? existingOwner)
    {
        if (existingOwner is not null && !string.Equals(existingOwner, scope.OwnerPrincipal, StringComparison.Ordinal))
            throw new SourceOwnershipException(source);
    }
}
