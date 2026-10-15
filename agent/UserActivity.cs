// Time since the last keyboard or mouse input in this Windows session.
// Only the time is read: no keys, no window titles, no screen.
using System.Runtime.InteropServices;

public static class UserActivity
{
    [StructLayout(LayoutKind.Sequential)]
    private struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }

    [DllImport("user32.dll")]
    private static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);

    // Null when not on Windows or when the call fails.
    public static double? SecondsSinceInput()
    {
        if (!OperatingSystem.IsWindows()) return null;
        var li = new LASTINPUTINFO { cbSize = (uint)Marshal.SizeOf<LASTINPUTINFO>() };
        if (!GetLastInputInfo(ref li)) return null;
        return unchecked((uint)Environment.TickCount - li.dwTime) / 1000.0;
    }
}
