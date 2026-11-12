using Rag.Core.Security;

namespace Rag.Core.Tests;

public class AccessScopeTests
{
    [Fact]
    public void ForUser_ListsUserAndGroupPrincipals()
    {
        var scope = AccessScope.ForUser(" alice ", ["hr", "  ", "eng", "hr"]);

        Assert.Equal("alice", scope.UserId);
        Assert.Equal(new[] { "user:alice", "group:hr", "group:eng" }, scope.Principals);
        Assert.Equal("user:alice", scope.OwnerPrincipal);
        Assert.False(scope.IsUnrestricted);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public void ForUser_BlankUserId_Throws(string userId) =>
        Assert.ThrowsAny<ArgumentException>(() => AccessScope.ForUser(userId));

    [Fact]
    public void Unrestricted_ReadsEverything_AndOwnsAsSystem()
    {
        var scope = AccessScope.Unrestricted;

        Assert.True(scope.IsUnrestricted);
        Assert.True(scope.CanRead(null));
        Assert.True(scope.CanRead([]));
        Assert.Equal(AccessScope.SystemOwnerPrincipal, scope.OwnerPrincipal);
    }

    [Fact]
    public void CanRead_RequiresAtLeastOneSharedPrincipal()
    {
        var scope = AccessScope.ForUser("alice", ["hr"]);

        Assert.True(scope.CanRead(["user:alice"]));
        Assert.True(scope.CanRead(["user:bob", "group:hr"]));
        Assert.False(scope.CanRead(["user:bob", "group:eng"]));
        Assert.False(scope.CanRead([]));
        Assert.False(scope.CanRead(null));
    }

    [Fact]
    public void CanRead_DoesNotTreatTheSystemOwnerAsAUser() =>
        Assert.False(AccessScope.ForUser("alice").CanRead([AccessScope.SystemOwnerPrincipal]));

    [Fact]
    public void BuildAcl_AlwaysIncludesOwnerFirst_AndDeduplicates()
    {
        var acl = AccessScope.ForUser("alice").BuildAcl(["group:hr", "user:alice", " group:hr ", "user:bob"]);

        Assert.Equal(new[] { "user:alice", "group:hr", "user:bob" }, acl);
    }

    [Fact]
    public void BuildAcl_WithoutSharing_IsOwnerOnly() =>
        Assert.Equal(new[] { "user:alice" }, AccessScope.ForUser("alice").BuildAcl(null));

    [Theory]
    [InlineData("everyone")]
    [InlineData("user:")]
    [InlineData("group:")]
    [InlineData("role:admin")]
    [InlineData("user: ")]
    [InlineData("user:bad\nname")]
    [InlineData("")]
    public void TryBuildAcl_InvalidPrincipal_Fails(string principal)
    {
        var ok = AccessScope.ForUser("alice").TryBuildAcl([principal], out _, out var error);

        Assert.False(ok);
        Assert.NotNull(error);
    }

    [Fact]
    public void TryBuildAcl_TooManyPrincipals_Fails()
    {
        var shared = Enumerable.Range(0, AccessScope.MaxPrincipals).Select(i => $"user:u{i}").ToList();

        var ok = AccessScope.ForUser("alice").TryBuildAcl(shared, out _, out var error);

        Assert.False(ok);
        Assert.NotNull(error);
    }

    [Fact]
    public void BuildAcl_InvalidPrincipal_ThrowsArgumentException() =>
        Assert.Throws<ArgumentException>(() => AccessScope.ForUser("alice").BuildAcl(["nonsense"]));

    [Theory]
    [InlineData("user:alice", true)]
    [InlineData("group:Human Resources", true)]
    [InlineData("user:" + "a", true)]
    [InlineData("user:alice\n", false)]
    [InlineData("USER:alice", false)]
    [InlineData(null, false)]
    public void IsValidPrincipal_ChecksShape(string? principal, bool expected) =>
        Assert.Equal(expected, AccessScope.IsValidPrincipal(principal));

    [Fact]
    public void IsValidPrincipal_RejectsOverlongNames() =>
        Assert.False(AccessScope.IsValidPrincipal("user:" + new string('a', AccessScope.MaxPrincipalLength + 1)));
}
