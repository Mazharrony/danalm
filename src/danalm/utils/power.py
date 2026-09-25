"""Windows power throttling (EcoQoS) for the current process (Phase 7, D-034).

Windows can run background processes at an efficiency setting; during D-033 that slowed the
unchanged Phase 5 model from 62.0 to 24.5 tokens per second. Measuring scripts turn it off for
their own process only; no system setting changes. Elsewhere this does nothing.
"""

import ctypes
import sys

PROCESS_POWER_THROTTLING = 4  # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
EXECUTION_SPEED = 0x1  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED


class _State(ctypes.Structure):
    _fields_ = [("version", ctypes.c_ulong), ("control_mask", ctypes.c_ulong),
                ("state_mask", ctypes.c_ulong)]  # fmt: skip


def disable_power_throttling() -> bool:
    """Opt this process out of EcoQoS. True if Windows accepted it, False elsewhere."""
    if sys.platform != "win32":
        return False
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                          wintypes.DWORD]  # fmt: skip
    k32.SetProcessInformation.restype = wintypes.BOOL
    state = _State(1, EXECUTION_SPEED, 0)  # control execution speed; not throttled
    return bool(k32.SetProcessInformation(k32.GetCurrentProcess(), PROCESS_POWER_THROTTLING,
                                          ctypes.byref(state), ctypes.sizeof(state)))  # fmt: skip
