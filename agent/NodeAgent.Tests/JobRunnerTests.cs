using System.Diagnostics;

namespace NodeAgent.Tests;

// Step 4b: the burn job must stop early when its token is cancelled.
public class JobRunnerTests
{
    [Fact]
    public void Already_cancelled_token_returns_at_once()
    {
        var cts = new CancellationTokenSource();
        cts.Cancel();
        var sw = Stopwatch.StartNew();
        JobRunner.Run(60, 1, cts.Token);
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(2), $"took {sw.Elapsed}");
    }

    [Fact]
    public void Cancel_during_run_stops_early()
    {
        var cts = new CancellationTokenSource();
        cts.CancelAfter(200);
        var sw = Stopwatch.StartNew();
        JobRunner.Run(60, 1, cts.Token);
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(3), $"took {sw.Elapsed}");
    }

    [Fact]
    public void No_token_runs_until_deadline()
    {
        var sw = Stopwatch.StartNew();
        JobRunner.Run(1, 1);
        Assert.True(sw.Elapsed >= TimeSpan.FromSeconds(0.9), $"took {sw.Elapsed}");
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(3), $"took {sw.Elapsed}");
    }
}
