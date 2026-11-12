using System.Globalization;
using System.Threading.RateLimiting;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.RateLimiting;
using Microsoft.Extensions.Options;

namespace Rag.Api.Security;

public static class SecurityExtensions
{
    public static IServiceCollection AddRagSecurity(
        this IServiceCollection services,
        IConfiguration configuration,
        IHostEnvironment environment)
    {
        services.Configure<AuthSettings>(configuration.GetSection(AuthSettings.SectionName));
        services.Configure<RateLimitSettings>(configuration.GetSection(RateLimitSettings.SectionName));

        var relaxed = environment.IsDevelopment() || environment.IsEnvironment("Testing");
        if (!relaxed && string.IsNullOrWhiteSpace(configuration[$"{AuthSettings.SectionName}:Authority"]))
            throw new InvalidOperationException(
                "Auth:Authority is not configured. Set it (and Auth:Audience) to your identity provider, " +
                "or run in the Development environment and use 'dotnet user-jwts' for local tokens.");

        services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme).AddJwtBearer();

        // Read the settings lazily so configuration overrides (tests, environment variables) always apply.
        services.AddOptions<JwtBearerOptions>(JwtBearerDefaults.AuthenticationScheme)
            .Configure<IOptions<AuthSettings>>((jwt, auth) =>
            {
                var settings = auth.Value;
                if (!string.IsNullOrWhiteSpace(settings.Authority))
                    jwt.Authority = settings.Authority;
                if (!string.IsNullOrWhiteSpace(settings.Audience))
                    jwt.Audience = settings.Audience;
                jwt.RequireHttpsMetadata = settings.RequireHttpsMetadata;
            });

        // Secure by default: every endpoint needs a signed-in user unless it opts out with AllowAnonymous.
        services.AddAuthorizationBuilder()
            .SetFallbackPolicy(new Microsoft.AspNetCore.Authorization.AuthorizationPolicyBuilder()
                .RequireAuthenticatedUser()
                .Build());

        services.AddRateLimiter(limiter =>
        {
            limiter.RejectionStatusCode = StatusCodes.Status429TooManyRequests;
            limiter.OnRejected = (context, _) =>
            {
                if (context.Lease.TryGetMetadata(MetadataName.RetryAfter, out var retryAfter))
                    context.HttpContext.Response.Headers.RetryAfter =
                        ((int)Math.Ceiling(retryAfter.TotalSeconds)).ToString(CultureInfo.InvariantCulture);
                return ValueTask.CompletedTask;
            };

            limiter.AddPolicy(RateLimitPolicies.Query, context =>
                PerUser(context, s => s.QueryPerMinute));
            limiter.AddPolicy(RateLimitPolicies.Upload, context =>
                PerUser(context, s => s.UploadPerMinute));
        });

        return services;
    }

    // One sliding window per user, so one heavy caller cannot use up the budget of the others.
    private static RateLimitPartition<string> PerUser(HttpContext context, Func<RateLimitSettings, int> limitOf)
    {
        var settings = context.RequestServices.GetRequiredService<IOptions<RateLimitSettings>>().Value;
        var user = context.User.FindFirst("sub")?.Value
                   ?? context.User.FindFirst(System.Security.Claims.ClaimTypes.NameIdentifier)?.Value
                   ?? "anonymous";

        return RateLimitPartition.GetSlidingWindowLimiter(user, _ => new SlidingWindowRateLimiterOptions
        {
            PermitLimit = Math.Max(1, limitOf(settings)),
            Window = TimeSpan.FromMinutes(1),
            SegmentsPerWindow = 6,
            QueueLimit = 0,
        });
    }
}
