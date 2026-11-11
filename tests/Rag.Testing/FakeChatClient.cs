using Microsoft.Extensions.AI;

namespace Rag.Testing;

/// <summary>Chat client that returns a scripted reply and records every request.</summary>
public sealed class FakeChatClient : IChatClient
{
    private readonly Func<IReadOnlyList<ChatMessage>, string> _reply;

    public FakeChatClient(string reply) : this(_ => reply) { }

    public FakeChatClient(Func<IReadOnlyList<ChatMessage>, string> reply) => _reply = reply;

    public List<IReadOnlyList<ChatMessage>> Requests { get; } = [];

    public int CallCount => Requests.Count;

    public IReadOnlyList<ChatMessage> LastRequest => Requests[^1];

    public Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var list = messages.ToList();
        Requests.Add(list);
        return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, _reply(list))));
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var response = await GetResponseAsync(messages, options, cancellationToken);
        foreach (var update in response.ToChatResponseUpdates())
            yield return update;
    }

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceKey is null && serviceType.IsInstanceOfType(this) ? this : null;

    public void Dispose() { }
}
