using System.Security.Claims;
using Rag.Core.Security;

namespace Rag.Api.Security;

public static class ClaimsPrincipalExtensions
{
    /// <summary>
    /// Builds the caller's <see cref="AccessScope"/> from the token: the <c>sub</c> claim (or the
    /// name identifier) is the user id and the configured claim lists the groups.
    /// Returns false when the token carries no usable user id.
    /// </summary>
    public static bool TryGetAccessScope(this ClaimsPrincipal principal, string groupClaim, out AccessScope scope)
    {
        scope = AccessScope.Unrestricted;

        var userId = principal.FindFirstValue("sub") ?? principal.FindFirstValue(ClaimTypes.NameIdentifier);
        if (string.IsNullOrWhiteSpace(userId) || !AccessScope.IsValidPrincipal(AccessScope.UserPrefix + userId.Trim()))
            return false;

        // Groups that would not make a valid principal (too long, control characters) are ignored.
        var groups = principal.FindAll(groupClaim)
            .Select(c => c.Value.Trim())
            .Where(g => g.Length > 0 && AccessScope.IsValidPrincipal(AccessScope.GroupPrefix + g))
            .Take(AccessScope.MaxPrincipals - 1);

        scope = AccessScope.ForUser(userId, groups);
        return true;
    }
}
