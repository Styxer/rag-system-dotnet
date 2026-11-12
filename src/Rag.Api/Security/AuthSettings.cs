namespace Rag.Api.Security;

/// <summary>Sign-in settings: which identity provider issues the JWTs this API accepts.</summary>
public sealed class AuthSettings
{
    public const string SectionName = "Auth";

    /// <summary>Issuer URL of the identity provider (Entra ID, Auth0, Keycloak…). Required outside Development.</summary>
    public string? Authority { get; set; }

    /// <summary>The audience (API identifier) tokens must be issued for.</summary>
    public string? Audience { get; set; }

    /// <summary>Name of the token claim that lists the caller's groups.</summary>
    public string GroupClaim { get; set; } = "groups";

    public bool RequireHttpsMetadata { get; set; } = true;
}

public sealed class RateLimitSettings
{
    public const string SectionName = "RateLimiting";

    /// <summary>Questions per user per minute (each one costs embedding and chat calls).</summary>
    public int QueryPerMinute { get; set; } = 30;

    /// <summary>Uploads per user per minute (each one costs embedding calls).</summary>
    public int UploadPerMinute { get; set; } = 10;
}

public static class RateLimitPolicies
{
    public const string Query = "query";
    public const string Upload = "upload";
}
