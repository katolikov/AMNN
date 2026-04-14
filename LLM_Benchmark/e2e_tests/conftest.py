"""
Shared fixtures and configuration for MNN Benchmark App E2E tests.
Uses uiautomator2 for Android UI automation.

Set DEVICE_SERIAL env var to target a specific device (e.g. R5CY71BJJ9D).
Defaults to the first available device.
"""

import os
import time
import subprocess
import pytest
import uiautomator2 as u2

APP_PACKAGE = "com.mnn.benchmarkapp"
MAIN_ACTIVITY = f"{APP_PACKAGE}/.MainActivity"
# Smallest model for fast download tests
TEST_MODEL_ID = "Qwen3-0.6B-MNN"
TEST_MODEL_NAME = "Qwen3 0.6B"
# Default path for ADB-pushed models
ADB_MODEL_PATH = "/data/local/tmp/mnn_bench/model"
# Timeouts
LAUNCH_TIMEOUT = 8
UI_TIMEOUT = 5
MODEL_LOAD_TIMEOUT = 120
MODEL_DOWNLOAD_TIMEOUT = 600
INFERENCE_TIMEOUT = 180

# Device serial — set via DEVICE_SERIAL env var or auto-detect
DEVICE_SERIAL = os.environ.get("DEVICE_SERIAL", "")


def _adb_cmd(*args):
    """Build an adb command list, inserting -s <serial> if a device is specified."""
    cmd = ["adb"]
    if DEVICE_SERIAL:
        cmd += ["-s", DEVICE_SERIAL]
    cmd += list(args)
    return cmd


def _keep_device_awake():
    """Prevent screen timeout, dismiss lock screen, and keep device awake."""
    try:
        # Set screen timeout to 30 minutes
        subprocess.run(
            _adb_cmd("shell", "settings", "put", "system",
                      "screen_off_timeout", "1800000"),
            capture_output=True, timeout=5
        )
        # Keep screen on while USB is connected
        subprocess.run(
            _adb_cmd("shell", "svc", "power", "stayon", "usb"),
            capture_output=True, timeout=5
        )
        # Turn screen on (in case it's off)
        subprocess.run(
            _adb_cmd("shell", "input", "keyevent", "KEYCODE_WAKEUP"),
            capture_output=True, timeout=5
        )
        time.sleep(0.5)
        # Dismiss lock screen with swipe up (Samsung lock screen)
        subprocess.run(
            _adb_cmd("shell", "input", "swipe", "540", "1800", "540", "600", "300"),
            capture_output=True, timeout=5
        )
        time.sleep(0.5)
        # Press HOME to dismiss any system UI overlay (notifications, etc.)
        subprocess.run(
            _adb_cmd("shell", "input", "keyevent", "KEYCODE_HOME"),
            capture_output=True, timeout=5
        )
        time.sleep(0.3)
        # Also try MENU to dismiss lock screen on older devices
        subprocess.run(
            _adb_cmd("shell", "input", "keyevent", "KEYCODE_MENU"),
            capture_output=True, timeout=5
        )
        time.sleep(0.3)
    except Exception:
        pass  # Best-effort


def _dismiss_permission_dialogs(device, max_attempts=5):
    """Handle any permission dialogs (camera, storage)."""
    for _ in range(max_attempts):
        allow_btn = device(text="While using the app")
        if allow_btn.exists:
            allow_btn.click()
            time.sleep(1)
            continue
        allow_btn2 = device(text="Allow")
        if allow_btn2.exists:
            allow_btn2.click()
            time.sleep(1)
            continue
        break


def _ensure_app_foreground(device, max_retries=5):
    """Ensure the app is in the foreground, retrying launch if needed."""
    for attempt in range(max_retries):
        current = device.app_current()
        if current["package"] == APP_PACKAGE:
            return True

        # App not in foreground — wake device, dismiss system UI, and relaunch
        _keep_device_awake()
        time.sleep(1)

        # Force-stop any blocking system UI
        if current["package"] == "com.android.systemui":
            subprocess.run(
                _adb_cmd("shell", "input", "keyevent", "KEYCODE_HOME"),
                capture_output=True, timeout=5
            )
            time.sleep(1)

        subprocess.run(
            _adb_cmd("shell", "am", "start", "-W", "-n", MAIN_ACTIVITY),
            capture_output=True, timeout=15
        )
        time.sleep(LAUNCH_TIMEOUT)
        _dismiss_permission_dialogs(device, max_attempts=3)
        time.sleep(1)

    current = device.app_current()
    return current["package"] == APP_PACKAGE


@pytest.fixture(scope="session")
def device():
    """Connect to the Android device/emulator via uiautomator2."""
    if DEVICE_SERIAL:
        d = u2.connect(DEVICE_SERIAL)
    else:
        d = u2.connect()
    d.implicitly_wait(UI_TIMEOUT)
    d.settings["wait_timeout"] = UI_TIMEOUT

    # Keep device awake for the entire test session
    _keep_device_awake()

    yield d


@pytest.fixture(autouse=True)
def launch_app(device):
    """Launch the app before each test and stop it after."""
    # Wake device before each test (prevents screen-off between tests)
    _keep_device_awake()

    device.app_stop(APP_PACKAGE)
    time.sleep(1)
    # Use adb am start directly — more reliable than uiautomator2 app_start
    subprocess.run(
        _adb_cmd("shell", "am", "start", "-n", MAIN_ACTIVITY),
        capture_output=True, timeout=10
    )
    time.sleep(LAUNCH_TIMEOUT)

    _dismiss_permission_dialogs(device)

    time.sleep(1)
    # Verify the app is in the foreground (with retry)
    assert _ensure_app_foreground(device), (
        f"App not in foreground after {3} retries. "
        f"Current: {device.app_current()['package']}"
    )
    yield
    device.app_stop(APP_PACKAGE)


def wait_for_text(device, text, timeout=UI_TIMEOUT):
    """Wait until a text element appears on screen."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if device(text=text).exists:
            return True
        time.sleep(0.5)
    return False


def wait_for_text_gone(device, text, timeout=UI_TIMEOUT):
    """Wait until a text element disappears from screen."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not device(text=text).exists:
            return True
        time.sleep(0.5)
    return False


def wait_for_content_desc(device, desc, timeout=UI_TIMEOUT):
    """Wait until an element with content-desc appears."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if device(description=desc).exists:
            return True
        time.sleep(0.5)
    return False


def get_all_texts(device):
    """Get all visible text elements on screen."""
    texts = []
    for elem in device.xpath("//*[@text!='']").all():
        t = elem.attrib.get("text", "")
        if t:
            texts.append(t)
    return texts


def get_all_content_descs(device):
    """Get all content-description values on screen."""
    descs = []
    for elem in device.xpath("//*[@content-desc!='']").all():
        d = elem.attrib.get("content-desc", "")
        if d:
            descs.append(d)
    return descs
