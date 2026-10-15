// Node Agent entry point. Joins the room for a Bearer token, then two loops:
//   telemetry loop: every 2s read sensors -> POST /ingest (identity from token)
//   job loop:       long-poll GET /jobs/next -> burn or chat LLM job
// All errors are logged and retried - the agent never crashes on a bad cycle.
// Never log password, token, or prompt text (docs/06).
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using NodeAgent;

var testMode = args.Contains("--test-sensors");
var testDl = args.Contains("--test-downloader");
var setupAt = Array.IndexOf(args, "--setup");
if (setupAt >= 0)
{
    var setupLink = setupAt + 1 < args.Length && !args[setupAt + 1].StartsWith("-")
        ? args[setupAt + 1] : null;
    var setupExit = AgentSetup.RunInteractive(setupLink);
    if (setupExit == 0)
    {
        // Wizard hoàn tất thì worker tự chạy outbound; không cần thao tác thứ hai.
        Process.Start(new ProcessStartInfo
        {
            FileName = Environment.ProcessPath!,
            UseShellExecute = true,
        });
    }
    Environment.Exit(setupExit);
}
var protectConfigAt = Array.IndexOf(args, "--protect-config");
if (protectConfigAt >= 0)
{
    if (protectConfigAt + 1 >= args.Length)
    {
        Console.Error.WriteLine("--protect-config cần đường dẫn config đích.");
        Environment.Exit(2);
    }
    var input = await JsonSerializer.DeserializeAsync<ProvisionInput>(
        Console.OpenStandardInput(), new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
        });
    if (input is null || string.IsNullOrWhiteSpace(input.NodeName)
        || string.IsNullOrWhiteSpace(input.ServerUrl)
        || string.IsNullOrWhiteSpace(input.RoomCode)
        || string.IsNullOrWhiteSpace(input.Password))
    {
        Console.Error.WriteLine("Dữ liệu provision thiếu trường bắt buộc.");
        Environment.Exit(2);
    }
    AgentConfig.WriteProtectedConfig(Path.GetFullPath(args[protectConfigAt + 1]),
        new AgentConfig(input.NodeName, input.ServerUrl, input.RoomCode,
            input.Password, input.LanUrl, input.TunnelUrl));
    Console.WriteLine("protected config written");
    Environment.Exit(0);
}
if (testDl)
Environment.Exit(await DownloaderSelfTest.RunAsync());

using var sensors = new SensorReader();

if (testMode)
{
    var s = sensors.Read();
    Console.WriteLine($"CPU temp: {s.CpuTemp?.ToString("F1") ?? "n/a"} °C");
    Console.WriteLine($"GPU temp: {s.GpuTemp?.ToString("F1") ?? "n/a"} °C");
    Console.WriteLine($"CPU util: {s.CpuUtil?.ToString("F0") ?? "n/a"} %");
    Console.WriteLine($"Power:    {s.PowerW?.ToString("F1") ?? "n/a"} W");
    Console.WriteLine("If temps show n/a: make sure you ran as Administrator.");
    return;
}

try
{
    Console.OutputEncoding = Encoding.UTF8;
    Console.InputEncoding = Encoding.UTF8;
}
catch { /* some hosts disallow console reconfigure */ }

var logPath = Path.Combine(AppContext.BaseDirectory, "agent.log");
void Log(string msg)
{
    var line = $"{DateTime.Now:HH:mm:ss} {msg}";
    Console.WriteLine(line);
    try
    {
        File.AppendAllText(logPath, line + Environment.NewLine, Encoding.UTF8);
    }
    catch { }
}
var configPath = AgentConfig.ResolveConfigPath();
var configReloadSync = new object();

DateTime ConfigWriteStamp()
{
    try
    {
        return File.Exists(configPath)
            ? File.GetLastWriteTimeUtc(configPath)
            : default;
    }
    catch (Exception)
    {
        return default;
    }
}

var configGate = new ConfigReloadGate(ConfigWriteStamp);
var cfg = AgentConfig.Load(Log);
var roster = new EndpointRoster(cfg.ServerUrl, msg => Log(msg));
var primaryUri = new Uri(cfg.ServerUrl.TrimEnd('/'));
if (!string.IsNullOrWhiteSpace(cfg.LanUrl)
    && roster.IsTrustedSecondary(cfg.LanUrl, primaryUri))
    roster.Add(cfg.LanUrl);
else if (!string.IsNullOrWhiteSpace(cfg.LanUrl))
    Log("[AGENT] bỏ LanUrl không tin cậy / HTTP ngoài loopback");
if (!string.IsNullOrWhiteSpace(cfg.TunnelUrl)
    && roster.IsTrustedSecondary(cfg.TunnelUrl, primaryUri))
    roster.Add(cfg.TunnelUrl);
else if (!string.IsNullOrWhiteSpace(cfg.TunnelUrl))
    Log("[AGENT] bỏ TunnelUrl không tin cậy / hạ HTTPS→HTTP");
if (roster.All.Count == 0)
{
    Log("[AGENT] roster rỗng sau validate — thoát");
    Environment.Exit(1);
}
var joinGate = new SemaphoreSlim(1, 1);
var rejoinGate = new SemaphoreSlim(1, 1);
// Backoff join khi 401/429 — tránh spam khóa rate-limit trên Host.
var joinBackoffMs = 2000;
const int JoinBackoffCapMs = 60_000;
long nextJoinAttemptAt = 0;

/// <summary>
/// The Host rewrites the protected config when its room is recreated: the room
/// code and the worker password both change while this process keeps running.
/// Reloading the file lets the worker join with the new pair instead of
/// replaying a secret the Host has already discarded, and it costs no second
/// elevation prompt. Returns true when the join credentials really changed.
/// </summary>
bool ReloadRotatedCredentials()
{
    lock (configReloadSync)
    {
        // Chỉ log loại lỗi, không log nội dung file hay bí mật. Cổng chỉ đẩy
        // mốc mtime sau khi nạp xong nên một lần lỗi vẫn còn cơ hội thử lại.
        if (!configGate.TryReload(() => AgentConfig.Load(Log), out var loaded,
                ex => Log($"[AGENT] could not reload config: {ex.GetType().Name}")))
            return false;
        var fresh = loaded!;
        var previous = cfg;
        cfg = fresh;
        if (previous.HasSameJoinCredentials(fresh))
            return false;
        if (!string.Equals(previous.ServerUrl, fresh.ServerUrl,
                StringComparison.Ordinal))
        {
            try
            {
                roster.ResetTo(fresh.ServerUrl, fresh.LanUrl, fresh.TunnelUrl);
            }
            catch (InvalidOperationException ex)
            {
                Log($"[AGENT] keeping previous endpoints: {ex.Message}");
            }
        }
        // A rotated credential is new information: the old backoff only counted
        // attempts against a room code and password that no longer exist.
        joinBackoffMs = 2000;
        Interlocked.Exchange(ref nextJoinAttemptAt, 0);
        Log("[AGENT] room credentials changed on disk — joining with the new pair");
        return true;
    }
}
Log($"[AGENT] {cfg.NodeName} starting, server = "
    + $"{EndpointRoster.SanitizeEndpointForLog(roster.Current)}, "
    + $"room = {cfg.RoomCode}");
if (roster.All.Count > 1)
    Log($"[AGENT] dual endpoint: {string.Join(" | ",
        roster.All.Select(EndpointRoster.SanitizeEndpointForLog))}");

int NextJoinDelayMs(System.Net.HttpStatusCode? status, int attempt)
{
    if (status is System.Net.HttpStatusCode.Unauthorized
        or System.Net.HttpStatusCode.TooManyRequests
        or System.Net.HttpStatusCode.Conflict)
    {
        var jitterMs = Random.Shared.Next(0, Math.Max(250, joinBackoffMs / 5));
        var delayMs = Math.Min(JoinBackoffCapMs, joinBackoffMs + jitterMs);
        Log($"[AGENT] join HTTP {(int)status} — backoff {delayMs}ms");
        joinBackoffMs = Math.Min(JoinBackoffCapMs, joinBackoffMs * 2);
        return delayMs;
    }
    return 2000 * Math.Max(1, attempt);
}

async Task DelayAfterJoinFailureAsync(System.Net.HttpStatusCode? status, int attempt)
{
    await Task.Delay(NextJoinDelayMs(status, attempt));
}

string? token = null;

async Task<HttpResponseMessage> SendFailoverAsync(
    Func<HttpClient, Task<HttpResponseMessage>> send, string? bearer = null)
    => await EndpointHttp.SendWithFailoverAsync(
        roster, send, bearer ?? token, Log);

var powerModel = PowerModelEstimator.TryLoad();
if (powerModel is not null)
    Log("[AGENT] power_model.json loaded — fallback energy_source=model when sensor silent");
else
    Log("[AGENT] no power_model.json — silent sensor → energy_source=none");

RoomConfig? roomCfg = null;
using var llm = new LlamaCppRunner(log: msg => Log(msg));

async Task<(string Token, RoomConfig Room)> JoinAsync()
{
    await joinGate.WaitAsync();
    try
    {
        ReloadRotatedCredentials();
        var body = JsonSerializer.Serialize(new Dictionary<string, object?>
        {
            ["room_code"] = cfg.RoomCode,
            ["password"] = cfg.Password,
            ["node_name"] = cfg.NodeName,
            ["role"] = "worker",
            ["capabilities"] = new Dictionary<string, object?>
            {
                ["cpu_cores"] = Environment.ProcessorCount,
                ["ram_gb"] = 0,
                ["os"] = Environment.OSVersion.ToString(),
                ["has_gpu"] = false,
                ["agent_version"] = "1.0.0-g4",
            },
        });
        using var resp = await SendFailoverAsync(
            h => h.PostAsync("/join",
                new StringContent(body, Encoding.UTF8, "application/json")),
            bearer: null);
        var text = await resp.Content.ReadAsStringAsync();
        if (!resp.IsSuccessStatusCode)
            throw new HttpRequestException(
                $"join failed HTTP {(int)resp.StatusCode}",
                null, resp.StatusCode);
        using var doc = JsonDocument.Parse(text);
        var newToken = doc.RootElement.GetProperty("token").GetString()
            ?? throw new InvalidOperationException("join response missing token");
        joinBackoffMs = 2000;
        Interlocked.Exchange(ref nextJoinAttemptAt, 0);
        // G6/H5: chỉ nhận URL tin cậy — pin theo ServerUrl / LAN / Cloudflare
        string? lan = null, tun = null;
        if (doc.RootElement.TryGetProperty("lan_url", out var lanEl))
            lan = lanEl.GetString();
        if (doc.RootElement.TryGetProperty("tunnel_url", out var tunEl))
            tun = tunEl.GetString();
        var primaryUri = new Uri(cfg.ServerUrl.TrimEnd('/'));
        if (lan is not null && !roster.IsTrustedSecondary(lan, primaryUri))
        {
            Log($"[AGENT] bỏ lan_url không tin cậy: "
                + EndpointRoster.SanitizeEndpointForLog(lan));
            lan = null;
        }
        if (tun is not null && !roster.IsTrustedSecondary(tun, primaryUri))
        {
            Log($"[AGENT] bỏ tunnel_url không tin cậy: "
                + EndpointRoster.SanitizeEndpointForLog(tun));
            tun = null;
        }
        roster.ResetTo(cfg.ServerUrl, lan, tun);
        cfg.SaveEndpoints(lan, tun);
        if (lan is not null || tun is not null)
            Log($"[AGENT] endpoints lan="
                + $"{(lan is null ? "–" : EndpointRoster.SanitizeEndpointForLog(lan))} "
                + $"tunnel={(tun is null ? "–" : EndpointRoster.SanitizeEndpointForLog(tun))}");
        var rc = RoomConfig.FromJson(doc.RootElement);
        Log("[AGENT] joined room as worker");
        return (newToken, rc);
    }
    finally
    {
        joinGate.Release();
    }
}

/// <summary>
/// Hai loop có thể cùng thấy token hết hạn. Chỉ request đầu tiên được join lại;
/// loop còn lại thấy token đã đổi thì dùng session mới, tránh bão 401/429.
/// </summary>
async Task<bool> RejoinAsync(string reason, string? rejectedToken)
{
    await rejoinGate.WaitAsync();
    try
    {
        if (!string.IsNullOrWhiteSpace(token)
            && !string.Equals(token, rejectedToken, StringComparison.Ordinal))
            return true;
        // Checked before the backoff gate: a long penalty must not hide the
        // credentials the Host has just written for this worker.
        ReloadRotatedCredentials();
        var now = Environment.TickCount64;
        if (now < Interlocked.Read(ref nextJoinAttemptAt))
            return false;
        token = null;
        try
        {
            var (newToken, rc) = await JoinAsync();
            token = newToken;
            await ApplyRoomConfigAsync(rc, reason);
            return true;
        }
        catch (Exception ex)
        {
            Log($"[AGENT] rejoin ({reason}) failed: {ex.Message}");
            var status = (ex as HttpRequestException)?.StatusCode;
            var delayMs = NextJoinDelayMs(status, 1);
            Interlocked.Exchange(ref nextJoinAttemptAt,
                Environment.TickCount64 + delayMs);
            return false;
        }
    }
    finally
    {
        rejoinGate.Release();
    }
}

// Nhiều lần hơn khi 409 NODE_NAME_TAKEN (chờ reclaim stale ~90s phía host).
for (var attempt = 1; attempt <= 12 && token is null; attempt++)
{
    try
    {
        var (t, rc) = await JoinAsync();
        token = t;
        roomCfg = rc;
    }
    catch (Exception ex)
    {
        Log($"[AGENT] join attempt {attempt} failed: {ex.Message}");
        var status = (ex as HttpRequestException)?.StatusCode;
        await DelayAfterJoinFailureAsync(status, attempt);
    }
}
if (token is null || roomCfg is null)
{
    Log("[AGENT] could not join room — exiting");
    Environment.Exit(1);
}

async Task LeaveBestEffortAsync()
{
    await LeaveAsync();
}

Console.CancelKeyPress += (_, e) =>
{
    e.Cancel = true;
    Log("[AGENT] shutdown — leave room");
    try { LeaveBestEffortAsync().GetAwaiter().GetResult(); }
    catch { /* best-effort */ }
    Environment.Exit(0);
};
AppDomain.CurrentDomain.ProcessExit += (_, _) =>
{
    try { LeaveBestEffortAsync().GetAwaiter().GetResult(); }
    catch { /* best-effort */ }
};

var llmReady = false;
var llmLock = new SemaphoreSlim(1, 1);
var modelSwitchBusy = 0;

bool RuntimeMatches(RoomConfig cfg) =>
    llm.IsRuntimeReady
    && llmReady
    && string.Equals(llm.ActiveModelSha256, cfg.ModelSha256,
        StringComparison.OrdinalIgnoreCase)
    && string.Equals(llm.ActiveRuntimeId, cfg.RuntimeId,
        StringComparison.OrdinalIgnoreCase)
    && llm.ActiveModelGeneration == cfg.ModelGeneration
    && (string.IsNullOrWhiteSpace(cfg.ModelId)
        || string.Equals(llm.ActiveModelId, cfg.ModelId,
            StringComparison.OrdinalIgnoreCase));

async Task ReportReadyAsync()
{
    if (roomCfg is null) return;
    try
    {
        using var resp = await SendFailoverAsync(
            h => h.PostAsync("/nodes/ready",
                new StringContent(
                    JsonSerializer.Serialize(new
                    {
                        model_id = roomCfg.ModelId,
                        model_sha256 = roomCfg.ModelSha256,
                        model_generation = roomCfg.ModelGeneration,
                        runtime_id = llmReady
                            ? llm.ActiveRuntimeId : roomCfg.RuntimeId,
                        runtime_ready = llmReady,
                    }),
                    Encoding.UTF8, "application/json")));
        _ = resp.StatusCode; // đọc status trước khi dispose
    }
    catch (Exception ex)
    {
        Log($"[AGENT] nodes/ready failed: {ex.Message}");
    }
}

async Task EnsureLlmAsync(string reason)
{
    if (roomCfg is null) return;
    // Báo false trước khi download/spawn để Host không dispatch nhầm runtime cũ.
    llmReady = false;
    await ReportReadyAsync();
    await llmLock.WaitAsync();
    try
    {
        var prog = new Progress<double>(p =>
        {
            if (p is 0 or 1 or >= 0.99)
                Log($"[LLM] download/startup progress {p:P0} ({reason})");
        });
        llmReady = await llm.EnsureReadyAsync(roomCfg, prog);
    }
    catch (DownloadRejectedException dex)
    {
        Log($"[LLM] supply-chain rejected:\n{dex.Message}");
        llmReady = false;
    }
    catch (Exception ex)
    {
        Log($"[LLM] not ready: {ex.Message} — continuing burn/telemetry");
        llmReady = false;
    }
    finally
    {
        llmLock.Release();
    }
    await ReportReadyAsync();
}

async Task ReportRuntimeDeathAsync()
{
    if (roomCfg is null || !llmReady || llm.IsRuntimeReady)
        return;
    // llama-server có thể chết khi nhàn rỗi. Hạ readiness trước chu kỳ
    // scheduler kế tiếp để Host không lease chat vào runtime đã mất.
    llmReady = false;
    Log("[LLM] runtime stopped outside a job — reporting not-ready");
    await ReportReadyAsync();
}

/// <summary>
/// Gán room_config mới; nếu pin model/hash khác runtime đang chạy → tải lại.
/// </summary>
async Task ApplyRoomConfigAsync(RoomConfig rc, string reason)
{
    var pinsChanged = roomCfg is null
        || !string.Equals(roomCfg.ModelId, rc.ModelId,
            StringComparison.OrdinalIgnoreCase)
        || !string.Equals(roomCfg.ModelSha256, rc.ModelSha256,
            StringComparison.OrdinalIgnoreCase)
        || roomCfg.ModelGeneration != rc.ModelGeneration;
    roomCfg = rc;
    if (pinsChanged)
    {
        llmReady = false;
        await ReportReadyAsync();
    }
    if (RuntimeMatches(rc))
        return;
    Log($"[LLM] pin đổi ({reason}) — EnsureReady "
        + $"id={rc.ModelId} hash={(rc.ModelSha256 ?? "?").Length}c");
    await EnsureLlmAsync(reason);
}

async Task LeaveAsync()
{
    try
    {
        using var resp = await SendFailoverAsync(
            h => h.PostAsync("/leave",
                new StringContent("{}", Encoding.UTF8, "application/json")));
        _ = resp.StatusCode;
    }
    catch (Exception ex)
    {
        Log($"[AGENT] leave failed: {ex.Message}");
    }
}

async Task SwitchModelIfNeededAsync()
{
    if (roomCfg is null || Volatile.Read(ref modelSwitchBusy) != 0) return;
    try
    {
        using var resp = await SendFailoverAsync(h => h.GetAsync("/api/state"));
        if (!resp.IsSuccessStatusCode) return;
        using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
        if (!doc.RootElement.TryGetProperty("llm_model_id", out var midEl))
            return;
        var remoteId = midEl.GetString();
        if (string.IsNullOrWhiteSpace(remoteId))
            return;

        // Id đã khớp nhưng runtime lệch hash / chưa ready → vẫn EnsureLlm.
        if (string.Equals(remoteId, roomCfg.ModelId,
                StringComparison.OrdinalIgnoreCase)
            && RuntimeMatches(roomCfg))
            return;

        if (Interlocked.Exchange(ref modelSwitchBusy, 1) != 0)
            return;

        // Tải model có thể mất vài phút. Đưa nó sang nền để telemetry vẫn
        // được gửi đều trong lúc runtime chưa sẵn sàng hoặc đang đổi model.
        _ = Task.Run(async () =>
        {
            try
            {
                if (!string.Equals(remoteId, roomCfg?.ModelId,
                        StringComparison.OrdinalIgnoreCase))
                {
                    Log($"[LLM] host đổi model → {remoteId} — leave/join + tải lại");
                    await LeaveAsync();
                    var (t, rc) = await JoinAsync();
                    token = t;
                    await ApplyRoomConfigAsync(rc, "model-switch");
                }
                else
                {
                    await EnsureLlmAsync("model-resync");
                }
            }
            catch (Exception ex)
            {
                Log($"[LLM] đổi model thất bại: {ex.Message}");
            }
            finally
            {
                Volatile.Write(ref modelSwitchBusy, 0);
            }
        });
    }
    catch (Exception ex)
    {
        Log($"[LLM] đổi model thất bại: {ex.Message}");
    }
}

// Telemetry trước — không chờ tải GGUF (UX demo).
var telemetryLoop = Task.Run(async () =>
{
    var cycle = 0;
    while (true)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(token))
            {
                await RejoinAsync("telemetry-session-missing", null);
                await Task.Delay(1000);
                continue;
            }
            await ReportRuntimeDeathAsync();
            var s = sensors.Read();
            double? powerW = s.PowerW;
            var powerSource = "none";
            if (powerW is not null)
                powerSource = "sensor";
            else if (powerModel is not null && s.CpuUtil is not null)
            {
                powerW = powerModel.EstimateW(s.CpuUtil, s.CpuTemp);
                if (powerW is not null)
                    powerSource = "model";
            }
            var body = JsonSerializer.Serialize(new Dictionary<string, object?>
            {
                ["cpu_temp"] = s.CpuTemp, ["gpu_temp"] = s.GpuTemp,
                ["cpu_util"] = s.CpuUtil, ["power_w"] = powerW,
                ["power_source"] = powerSource,
                ["cpu_clock_mhz"] = s.CpuClockMhz,
                ["fan_rpm"] = s.FanRpm,
                ["seconds_since_input"] = UserActivity.SecondsSinceInput(),
            });
            using (var resp = await SendFailoverAsync(
                h => h.PostAsync("/ingest",
                    new StringContent(body, Encoding.UTF8, "application/json")),
                token))
            {
                if (resp.StatusCode is System.Net.HttpStatusCode.Unauthorized
                    or System.Net.HttpStatusCode.Forbidden)
                {
                    Log("[TELEMETRY] token rejected — rejoining");
                    await RejoinAsync("telemetry-rejoin", token);
                }
                else if (!resp.IsSuccessStatusCode)
                    Log($"[TELEMETRY] server said {(int)resp.StatusCode}");
            }

            // Mỗi ~10s kiểm tra admin có đổi model không.
            if (++cycle % 5 == 0)
                await SwitchModelIfNeededAsync();
        }
        catch (Exception ex)
        {
            Log($"[TELEMETRY] send failed: {ex.Message} (is the server up? firewall?)");
        }
        await Task.Delay(2000);
    }
});

// LLM tải song song với telemetry; báo ready khi xong.
_ = Task.Run(async () => await EnsureLlmAsync("startup"));

async Task<bool> PostAgentPayloadAsync(string path, object payload)
{
    var json = JsonSerializer.Serialize(payload);
    for (var attempt = 0; attempt < 3; attempt++)
    {
        using var resp = await SendFailoverAsync(
            h => h.PostAsync(path,
                new StringContent(json, Encoding.UTF8, "application/json")), token);
        if (resp.IsSuccessStatusCode) return true;
        var body = "";
        try { body = await resp.Content.ReadAsStringAsync(); }
        catch { /* ignore read errors */ }
        if (body.Length > 200) body = body[..200];
        if (resp.StatusCode is System.Net.HttpStatusCode.Unauthorized
            or System.Net.HttpStatusCode.Forbidden)
        {
            Log($"[JOB] {path} HTTP {(int)resp.StatusCode} — rejoining");
            await RejoinAsync("job-post-rejoin", token);
            continue;
        }
        if (resp.StatusCode == System.Net.HttpStatusCode.Conflict)
        {
            Log($"[JOB] {path} HTTP 409 stale attempt body={body}");
            return false;
        }
        var retryAfterMs = 250 * (attempt + 1);
        if (resp.StatusCode == System.Net.HttpStatusCode.TooManyRequests
            && resp.Headers.RetryAfter?.Delta is TimeSpan ra)
            retryAfterMs = (int)Math.Clamp(ra.TotalMilliseconds, 250, 60_000);
        Log($"[JOB] {path} HTTP {(int)resp.StatusCode} retry={attempt + 1} "
            + $"body={body}");
        await Task.Delay(retryAfterMs);
    }
    Log($"[JOB] {path} failed after retries");
    return false;
}

async Task PostResultAsync(Dictionary<string, object?> result, string? attemptId)
{
    if (!string.IsNullOrWhiteSpace(attemptId))
        result["attempt_id"] = attemptId;
    await PostAgentPayloadAsync("/jobs/result", result);
}

async Task<int> PostChunkAsync(string jobId, string? attemptId, int firstSeq,
    string delta)
{
    if (string.IsNullOrWhiteSpace(attemptId) || string.IsNullOrEmpty(delta))
        return 0;
    var sent = 0;
    foreach (var piece in delta.Chunk(4096))
    {
        var ok = await PostAgentPayloadAsync(
            $"/jobs/{Uri.EscapeDataString(jobId)}/events",
            new { attempt_id = attemptId, seq = firstSeq + sent,
                delta = new string(piece) });
        if (!ok) return sent;
        sent++;
    }
    return sent;
}

var jobLoop = Task.Run(async () =>
{
    while (true)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(token))
            {
                await RejoinAsync("job-session-missing", null);
                await Task.Delay(1000);
                continue;
            }
            using var resp = await SendFailoverAsync(
                h => h.GetAsync("/jobs/next?wait=25"), token);
            if (resp.StatusCode == System.Net.HttpStatusCode.OK)
            {
                using var doc = JsonDocument.Parse(
                    await resp.Content.ReadAsStringAsync());
                var root = doc.RootElement;
                var id = root.GetProperty("id").GetString() ?? "?";
                var attemptId = root.TryGetProperty("attempt_id", out var attemptEl)
                    ? attemptEl.GetString() : null;
                var type = root.TryGetProperty("type", out var tEl)
                    ? tEl.GetString() ?? "burn" : "burn";

                if (type == "chat")
                {
                    if (!llmReady || roomCfg is null || !RuntimeMatches(roomCfg))
                    {
                        if (llmReady)
                        {
                            llmReady = false;
                            await ReportReadyAsync();
                        }
                        Log($"[JOB] {id} chat but runtime not ready → error");
                        await PostResultAsync(new Dictionary<string, object?>
                        {
                            ["job_id"] = id,
                            ["status"] = "error",
                            ["error_message"] = "llm runtime not ready",
                            ["energy_source"] = "none",
                        }, attemptId);
                        continue;
                    }

                    var prompt = root.GetProperty("prompt").GetString() ?? "";
                    var maxTokens = 512;
                    var temperature = 0.7;
                    var wantStream = false;
                    if (root.TryGetProperty("params", out var pEl))
                    {
                        if (pEl.TryGetProperty("max_tokens", out var mt)
                            && mt.TryGetInt32(out var mtRaw))
                        {
                            if (!JobParamGuard.TryClampMaxTokens(
                                    mtRaw, out maxTokens, out var mtReason))
                                Log($"[JOB] {id} clamp {mtReason} → {maxTokens}");
                        }
                        if (pEl.TryGetProperty("temperature", out var tp)
                            && tp.TryGetDouble(out var tpRaw))
                        {
                            if (!JobParamGuard.TryClampTemperature(
                                    tpRaw, out temperature, out var tpReason))
                                Log($"[JOB] {id} clamp {tpReason} → {temperature}");
                        }
                        if (pEl.TryGetProperty("stream", out var stEl)
                            && (stEl.ValueKind is JsonValueKind.True
                                or JsonValueKind.False))
                            wantStream = stEl.GetBoolean();
                    }
                    var deadlineS = 60;
                    if (root.TryGetProperty("deadline_s", out var dl)
                        && dl.TryGetInt32(out var dlRaw))
                    {
                        if (!JobParamGuard.TryClampDeadlineS(
                                dlRaw, out deadlineS, out var dlReason))
                            Log($"[JOB] {id} clamp {dlReason} → {deadlineS}");
                    }
                    // Log metadata only — NEVER prompt text (S13).
                    Log($"[JOB] chat {id}: max_tokens={maxTokens} "
                        + $"prompt_len={prompt.Length} deadline={deadlineS}s "
                        + $"stream={wantStream}");

                    var energy = new EnergySampler(powerModel);
                    double? peakTemp = null;
                    double? minClock = null;
                    double? lastFan = null;
                    using var cts = new CancellationTokenSource(
                        TimeSpan.FromSeconds(Math.Max(5, deadlineS)));
                    var sw = Stopwatch.StartNew();
                    try
                    {
                        var sampleTask = Task.Run(async () =>
                        {
                            while (!cts.IsCancellationRequested)
                            {
                                var s = sensors.Read();
                                energy.Sample(s.PowerW, s.CpuUtil, s.CpuTemp);
                                if (s.CpuTemp is double ct)
                                    peakTemp = peakTemp is null
                                        ? ct : Math.Max(peakTemp.Value, ct);
                                if (s.CpuClockMhz is double clk)
                                    minClock = minClock is null
                                        ? clk : Math.Min(minClock.Value, clk);
                                if (s.FanRpm is double fr)
                                    lastFan = fr;
                                try { await Task.Delay(200, cts.Token); }
                                catch (OperationCanceledException) { break; }
                            }
                        }, cts.Token);

                        EventBatcher? batcher = wantStream
                            ? new EventBatcher(id, attemptId, 0, PostChunkAsync)
                            : null;
                        Func<string, Task>? onDelta = batcher is null
                            ? null
                            : async delta => await batcher.AddAsync(delta);
                        var result = await llm.GenerateAsync(
                            prompt, new GenParams(maxTokens, temperature),
                            cts.Token, onDelta);
                        if (batcher is not null)
                            await batcher.FlushAsync();
                        cts.Cancel();
                        try { await sampleTask; } catch { /* cancelled */ }
                        sw.Stop();
                        var (ej, esrc) = energy.Finalize();
                        Log($"[JOB] {id} done tokens_out={result.TokensOut} "
                            + $"duration_ms={sw.Elapsed.TotalMilliseconds:F0} "
                            + $"energy_source={esrc}"
                            + (result.TokensEstimated ? " tokens_estimated" : ""));
                        await PostResultAsync(new Dictionary<string, object?>
                        {
                            ["job_id"] = id,
                            ["status"] = "ok",
                            ["text"] = result.Text,
                            ["tokens_in"] = result.TokensIn,
                            ["tokens_out"] = result.TokensOut,
                            ["tokens_estimated"] = result.TokensEstimated
                                || result.TokensOut is null,
                            ["duration_ms"] = sw.Elapsed.TotalMilliseconds,
                            ["prompt_eval_ms"] = result.PromptMs,
                            ["energy_j"] = ej,
                            ["energy_source"] = esrc,
                            ["peak_temp_c"] = peakTemp,
                            ["min_clock_mhz"] = minClock,
                            ["fan_rpm"] = lastFan,
                            ["cpu_util"] = sensors.Read().CpuUtil,
                        }, attemptId);
                    }
                    catch (OperationCanceledException)
                    {
                        sw.Stop();
                        Log($"[JOB] {id} TIMEOUT — restart runtime (X10)");
                        try
                        {
                            if (roomCfg is not null)
                                await llm.RestartAsync(roomCfg);
                        }
                        catch (Exception rex)
                        {
                            Log($"[LLM] restart failed: {rex.Message}");
                            llmReady = false;
                        }
                        var (ej, esrc) = energy.Finalize();
                        await PostResultAsync(new Dictionary<string, object?>
                        {
                            ["job_id"] = id,
                            ["status"] = "timeout",
                            ["error_message"] = "llm job deadline exceeded",
                            ["duration_ms"] = sw.Elapsed.TotalMilliseconds,
                            ["energy_j"] = ej,
                            ["energy_source"] = esrc,
                            ["peak_temp_c"] = peakTemp,
                        }, attemptId);
                    }
                    catch (Exception ex)
                    {
                        if (!llm.IsRuntimeReady)
                        {
                            llmReady = false;
                            await ReportReadyAsync();
                        }
                        Log($"[JOB] {id} chat error: {ex.Message}");
                        await PostResultAsync(new Dictionary<string, object?>
                        {
                            ["job_id"] = id,
                            ["status"] = "error",
                            ["error_message"] = ex.Message.Length > 512
                                ? ex.Message[..512] : ex.Message,
                            ["energy_source"] = "none",
                        }, attemptId);
                    }
                    continue;
                }

                // burn job (demo / calibrate)
                int duration = 10;
                if (root.TryGetProperty("duration_s", out var dEl)
                    && dEl.TryGetInt32(out var dRaw))
                {
                    if (!JobParamGuard.TryClampDurationS(
                            dRaw, out duration, out var dReason))
                        Log($"[JOB] {id} clamp {dReason} → {duration}");
                }
                int cores = 0;
                if (root.TryGetProperty("cores", out var cEl)
                    && cEl.TryGetInt32(out var cRaw))
                {
                    if (!JobParamGuard.TryClampCores(
                            cRaw, Environment.ProcessorCount,
                            out cores, out var cReason))
                        Log($"[JOB] {id} clamp {cReason} → {cores}");
                }
                Log($"[JOB] burn {id}: {duration}s on "
                    + $"{(cores == 0 ? "all" : cores.ToString())} cores");
                var burnEnergy = new EnergySampler(powerModel);
                var burnEnd = DateTime.UtcNow.AddSeconds(duration);
                using var burnCts = new CancellationTokenSource();
                var burnStart = DateTime.UtcNow;
                var burnTask = Task.Run(
                    () => JobRunner.Run(duration, cores, burnCts.Token));
                double? burnMinClock = null;
                var preempted = false;
                while (DateTime.UtcNow < burnEnd && !burnTask.IsCompleted)
                {
                    // Input that happened after the job started means a person
                    // is back at this PC: stop the burn and give the PC back.
                    if (UserActivity.SecondsSinceInput() is double si
                        && si < (DateTime.UtcNow - burnStart).TotalSeconds)
                    {
                        preempted = true;
                        burnCts.Cancel();
                        Log($"[JOB] {id} preempted: user input on this PC");
                        break;
                    }
                    var bs = sensors.Read();
                    burnEnergy.Sample(bs.PowerW, bs.CpuUtil, bs.CpuTemp);
                    if (bs.CpuClockMhz is double clk)
                        burnMinClock = burnMinClock is null
                            ? clk : Math.Min(burnMinClock.Value, clk);
                    await Task.Delay(200);
                }
                await burnTask;
                var (bej, bes) = burnEnergy.Finalize();
                Log($"[JOB] {id} done energy_source={bes} preempted={preempted}");
                await PostResultAsync(new Dictionary<string, object?>
                {
                    ["job_id"] = id,
                    ["status"] = preempted ? "preempted" : "ok",
                    ["energy_j"] = bej,
                    ["energy_source"] = bes,
                    ["min_clock_mhz"] = burnMinClock,
                }, attemptId);
                continue;
            }
            if (resp.StatusCode is System.Net.HttpStatusCode.Unauthorized
                or System.Net.HttpStatusCode.Forbidden)
            {
                Log("[JOB] token rejected — rejoining");
                await RejoinAsync("job-rejoin", token);
            }
        }
        catch (Exception ex)
        {
            Log($"[JOB] poll failed: {ex.Message}");
        }
        await Task.Delay(1000);
    }
});

await Task.WhenAll(telemetryLoop, jobLoop);

record ProvisionInput(string NodeName, string ServerUrl, string RoomCode,
    string Password, string? LanUrl = null, string? TunnelUrl = null);
