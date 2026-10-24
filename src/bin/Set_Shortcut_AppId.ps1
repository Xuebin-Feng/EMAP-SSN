# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
#
# Give a Windows shortcut (.lnk) an AppUserModelID, so the taskbar groups the windows of the process that
# sets the same ID (Desktop_App.set_app_user_model_id) under the shortcut's name and icon, and a pinned
# shortcut finds its window again. Used by install.bat; plain PowerShell, no extra packages.
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File Set_Shortcut_AppId.ps1 -Lnk <file.lnk> -AppId <id>
#
# It touches only the shortcut it is given, and exits with 1 if it could not set the ID.
param(
    [Parameter(Mandatory = $true)][string]$Lnk,
    [Parameter(Mandatory = $true)][string]$AppId
)

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class ShortcutAppId
{
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct PropertyKey { public Guid fmtid; public uint pid; }

    [StructLayout(LayoutKind.Explicit)]
    public struct PropVariant
    {
        [FieldOffset(0)] public ushort vt;
        [FieldOffset(8)] public IntPtr pointerValue;
    }

    [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")]
    public interface IPropertyStore
    {
        uint GetCount(out uint count);
        uint GetAt(uint index, out PropertyKey key);
        uint GetValue(ref PropertyKey key, out PropVariant value);
        uint SetValue(ref PropertyKey key, ref PropVariant value);
        uint Commit();
    }

    [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
    static extern void SHGetPropertyStoreFromParsingName(string path, IntPtr bindContext, int flags, ref Guid iid, out IPropertyStore store);

    public static void Set(string path, string appId)
    {
        Guid iid = new Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99");
        IPropertyStore store;
        SHGetPropertyStoreFromParsingName(path, IntPtr.Zero, 2 /* GPS_READWRITE */, ref iid, out store);
        PropertyKey key = new PropertyKey { fmtid = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), pid = 5 }; // System.AppUserModel.ID
        PropVariant value = new PropVariant { vt = 31 /* VT_LPWSTR */, pointerValue = Marshal.StringToCoTaskMemUni(appId) };
        store.SetValue(ref key, ref value);
        store.Commit();
        Marshal.FreeCoTaskMem(value.pointerValue);
        Marshal.ReleaseComObject(store);
    }
}
"@

try {
    [ShortcutAppId]::Set((Resolve-Path -LiteralPath $Lnk).Path, $AppId)
} catch {
    Write-Host "[INFO] Could not set the AppUserModelID on $Lnk ($($_.Exception.Message)); the shortcut works without it."
    exit 1
}
