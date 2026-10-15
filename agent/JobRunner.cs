// Executes CPU-burn jobs: pure math loops on N threads until the deadline.
// This is the "workload" being balanced - it genuinely heats the CPU.
public static class JobRunner
{
    // ct lets the agent stop the burn early (Step 4b: a person came back).
    public static void Run(int durationS, int cores,
                           CancellationToken ct = default)
    {
        if (cores <= 0 || cores > Environment.ProcessorCount)
            cores = Environment.ProcessorCount;
        var end = DateTime.UtcNow.AddSeconds(durationS);
        Parallel.For(0, cores, _ =>
        {
            double x = 1.0001;
            while (DateTime.UtcNow < end && !ct.IsCancellationRequested)
                x = Math.Sqrt(x * 1.0001) + 0.0001;
        });
    }
}
