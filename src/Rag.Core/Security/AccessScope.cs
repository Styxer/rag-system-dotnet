using System.Text.RegularExpressions;

namespace Rag.Core.Security;

/// <summary>
/// Who is asking: the caller's identity as a set of principals (<c>user:{id}</c> plus one
/// <c>group:{name}</c> per group). Every chunk stored in the vector store carries an access list of
/// principals, and a search only returns chunks whose list shares at least one principal with the caller.
/// Every store and service method takes a scope explicitly, so access control cannot be forgotten.
/// </summary>
public sealed partial class AccessScope
{
    public const string UserPrefix = "user:";
    public const string GroupPrefix = "group:";

    /// <summary>
    /// Owner of data created without a user (seeding, tests, rows from before access control existed).
    /// It does not start with "user:", so no real caller can ever have it.
    /// </summary>
    public const string SystemOwnerPrincipal = "system:internal";

    public const int MaxPrincipals = 50;
    public const int MaxPrincipalLength = 200;

    private readonly HashSet<string> _principalSet;

    private AccessScope(string userId, IReadOnlyList<string> principals, bool isUnrestricted)
    {
        UserId = userId;
        Principals = principals;
        IsUnrestricted = isUnrestricted;
        _principalSet = new HashSet<string>(principals, StringComparer.Ordinal);
    }

    /// <summary>Sees and manages everything. For trusted internal code and tests, never for request handling.</summary>
    public static AccessScope Unrestricted { get; } = new("system", [SystemOwnerPrincipal], isUnrestricted: true);

    public string UserId { get; }

    /// <summary>The caller's principals; the first one is always <c>user:{UserId}</c>.</summary>
    public IReadOnlyList<string> Principals { get; }

    public bool IsUnrestricted { get; }

    /// <summary>The principal recorded as owner of documents this caller uploads.</summary>
    public string OwnerPrincipal => IsUnrestricted ? SystemOwnerPrincipal : UserPrefix + UserId;

    public static AccessScope ForUser(string userId, IEnumerable<string>? groups = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(userId);

        var id = userId.Trim();
        var principals = new List<string> { UserPrefix + id };
        foreach (var group in groups ?? Array.Empty<string>())
        {
            if (!string.IsNullOrWhiteSpace(group))
                principals.Add(GroupPrefix + group.Trim());
        }

        return new AccessScope(id, principals.Distinct(StringComparer.Ordinal).ToList(), isUnrestricted: false);
    }

    /// <summary>True when the caller may read a chunk with this access list.</summary>
    public bool CanRead(IReadOnlyList<string>? allowedPrincipals)
    {
        if (IsUnrestricted)
            return true;
        if (allowedPrincipals is null)
            return false;

        foreach (var principal in allowedPrincipals)
        {
            if (_principalSet.Contains(principal))
                return true;
        }
        return false;
    }

    /// <summary>
    /// Builds the access list for a document: the owner always has access, plus the principals the
    /// owner shares it with. Throws <see cref="ArgumentException"/> for invalid input.
    /// </summary>
    public IReadOnlyList<string> BuildAcl(IEnumerable<string>? sharedWith)
    {
        if (!TryBuildAcl(sharedWith, out var acl, out var error))
            throw new ArgumentException(error, nameof(sharedWith));
        return acl;
    }

    public bool TryBuildAcl(IEnumerable<string>? sharedWith, out IReadOnlyList<string> acl, out string? error)
    {
        var result = new List<string> { OwnerPrincipal };

        foreach (var raw in sharedWith ?? Array.Empty<string>())
        {
            var principal = raw?.Trim() ?? string.Empty;
            if (!IsValidPrincipal(principal))
            {
                acl = result;
                error = $"Each principal must look like 'user:<id>' or 'group:<name>' (up to {MaxPrincipalLength} characters, no control characters).";
                return false;
            }

            if (!result.Contains(principal))
                result.Add(principal);

            if (result.Count > MaxPrincipals)
            {
                acl = result;
                error = $"A document can be shared with at most {MaxPrincipals} principals.";
                return false;
            }
        }

        acl = result;
        error = null;
        return true;
    }

    public static bool IsValidPrincipal(string? principal) =>
        principal is not null && PrincipalRegex().IsMatch(principal);

    // \z (not $) so that a trailing newline cannot slip through.
    [GeneratedRegex(@"^(user|group):[^\p{C}\s][^\p{C}]{0,199}\z")]
    private static partial Regex PrincipalRegex();
}
