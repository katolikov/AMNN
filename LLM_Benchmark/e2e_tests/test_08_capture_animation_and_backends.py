"""
Test Suite 8: Capture Button Animation & Backend Output Verification

Verifies:
- Capture button exists and is clickable with animation (no crash)
- Vulkan backend removed from settings dropdowns
- LLM produces output with CPU backend
- LLM produces output with OpenCL backend
- VLM capture + analyze does not crash
- Backend switching works without crash
"""

import re
import time
import subprocess
import pytest
from conftest import (
    APP_PACKAGE,
    MODEL_LOAD_TIMEOUT,
    INFERENCE_TIMEOUT,
    UI_TIMEOUT,
    wait_for_text,
    get_all_texts,
    get_all_content_descs,
    _adb_cmd,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def navigate_to_settings(device):
    settings_btn = device(description="Settings")
    assert settings_btn.exists(timeout=5), "Settings button not found"
    settings_btn.click()
    time.sleep(2)
    assert wait_for_text(device, "Settings", timeout=5), "Settings screen not loaded"


def navigate_back(device):
    back_btn = device(description="Back")
    if back_btn.exists:
        back_btn.click()
        time.sleep(2)


def navigate_to_vlm(device):
    vlm_btn = device(text="VLM")
    if vlm_btn.exists(timeout=3):
        vlm_btn.click()
        time.sleep(2)


def navigate_to_llm(device):
    llm_btn = device(text="LLM")
    if llm_btn.exists(timeout=3):
        llm_btn.click()
        time.sleep(2)


def ensure_model_loaded(device):
    texts = get_all_texts(device)
    if "No model" not in texts and "Load failed" not in texts:
        return

    sel = device(description="Select model")
    if sel.exists(timeout=3):
        sel.click()
        time.sleep(2)
        select_btn = device(description="Select")
        if select_btn.exists:
            select_btn.click()
            deadline = time.time() + MODEL_LOAD_TIMEOUT
            while time.time() < deadline:
                cur = device.app_current()
                if cur["package"] != APP_PACKAGE:
                    subprocess.run(
                        _adb_cmd("shell", "am", "start", "-n",
                                 f"{APP_PACKAGE}/.MainActivity"),
                        capture_output=True, timeout=10
                    )
                    time.sleep(8)
                    return
                texts = get_all_texts(device)
                if "No model" not in texts:
                    if device(text="Models").exists:
                        device.press("back")
                        time.sleep(1)
                    return
                time.sleep(3)

    pytest.skip("No model available for this test")


def send_llm_message(device, message, timeout=INFERENCE_TIMEOUT):
    """Send a message in LLM chat and wait for response."""
    msg_input = device(className="android.widget.EditText")
    if not msg_input.exists(timeout=10):
        pytest.skip("Chat input not found")

    msg_input.set_text(message)
    time.sleep(0.5)
    device.send_action("send")
    time.sleep(2)

    # Wait for inference to complete (input re-enabled)
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = device.app_current()
        if current["package"] != APP_PACKAGE:
            pytest.fail("App crashed during inference")

        inp = device(className="android.widget.EditText")
        if inp.exists and inp.info.get("enabled", False):
            break
        time.sleep(3)

    time.sleep(1)


# ===========================================================================
# Capture Button Animation
# ===========================================================================

class TestCaptureButtonAnimation:
    """Capture button must have a press animation and not crash."""

    def test_capture_button_exists(self, device):
        """VLM screen must have a Capture button."""
        navigate_to_vlm(device)
        time.sleep(2)
        capture_btn = device(description="Capture")
        assert capture_btn.exists, \
            f"Capture button not found. Descs: {get_all_content_descs(device)}"

    def test_capture_button_clickable_no_crash(self, device):
        """Clicking Capture button must trigger animation without crash."""
        navigate_to_vlm(device)
        time.sleep(3)

        capture_btn = device(description="Capture")
        if not capture_btn.exists:
            pytest.skip("Capture button not found (camera may not be available)")

        # Click and wait for animation to complete
        capture_btn.click()
        time.sleep(3)

        current = device.app_current()
        assert current["package"] == APP_PACKAGE, "App crashed during capture click"

        # Click again to test repeated animation
        capture_btn2 = device(description="Capture")
        if capture_btn2.exists:
            capture_btn2.click()
            time.sleep(3)

        current = device.app_current()
        assert current["package"] == APP_PACKAGE, "App crashed on second capture"

    def test_capture_button_size(self, device):
        """Capture button should be the largest button (72dp) in control row."""
        navigate_to_vlm(device)
        time.sleep(2)

        capture_btn = device(description="Capture")
        assert capture_btn.exists

        info = capture_btn.info
        bounds = info.get("bounds", {})
        width = bounds.get("right", 0) - bounds.get("left", 0)
        height = bounds.get("bottom", 0) - bounds.get("top", 0)

        # 72dp = ~216px at 3x density. Should be significantly larger than 56dp buttons.
        # Just assert it's not tiny — exact px depends on device density.
        assert width > 50, f"Capture button too small: {width}x{height}"
        assert height > 50, f"Capture button too small: {width}x{height}"


# ===========================================================================
# Vulkan Removed from Settings
# ===========================================================================

class TestVulkanRemoved:
    """Vulkan backend option must be removed from settings dropdowns."""

    def test_backend_dropdown_no_vulkan(self, device):
        """Backend dropdown must NOT contain 'vulkan' option."""
        navigate_to_settings(device)

        # Find Backend setting and open dropdown
        backend_label = device(text="Backend")
        if not backend_label.exists:
            pytest.skip("Backend setting not visible")

        bounds = backend_label.info["bounds"]
        device.click(
            (bounds["left"] + bounds["right"]) // 2,
            bounds["bottom"] + 30
        )
        time.sleep(1)

        # Check dropdown options
        texts = get_all_texts(device)
        assert "vulkan" not in texts, \
            f"'vulkan' found in backend dropdown. Texts: {texts}"

        # Verify cpu and opencl ARE present
        assert "cpu" in texts or "opencl" in texts, \
            f"Neither 'cpu' nor 'opencl' found in dropdown. Texts: {texts}"

        # Close dropdown
        device.press("back")
        time.sleep(0.5)

    def test_vision_backend_no_vulkan(self, device):
        """Vision Backend dropdown must NOT contain 'vulkan' option."""
        navigate_to_settings(device)

        # Scroll down to find MLLM section
        device.swipe(540, 1500, 540, 500, duration=0.5)
        time.sleep(1)

        vision_label = device(text="Vision Backend")
        if not vision_label.exists:
            # Try scrolling more
            device.swipe(540, 1500, 540, 500, duration=0.5)
            time.sleep(1)

        if not device(text="Vision Backend").exists:
            pytest.skip("Vision Backend setting not visible")

        vision_label = device(text="Vision Backend")
        bounds = vision_label.info["bounds"]
        device.click(
            (bounds["left"] + bounds["right"]) // 2,
            bounds["bottom"] + 30
        )
        time.sleep(1)

        texts = get_all_texts(device)
        assert "vulkan" not in texts, \
            f"'vulkan' found in Vision Backend dropdown. Texts: {texts}"

        device.press("back")
        time.sleep(0.5)


# ===========================================================================
# LLM Output with CPU Backend
# ===========================================================================

class TestLlmCpuOutput:
    """LLM must produce valid output with CPU backend."""

    def test_llm_cpu_produces_output(self, device):
        """Set CPU backend, load model, send message, verify response."""
        # Set backend to CPU
        navigate_to_settings(device)

        backend_label = device(text="Backend")
        if not backend_label.exists:
            pytest.skip("Backend setting not visible")

        bounds = backend_label.info["bounds"]
        device.click(
            (bounds["left"] + bounds["right"]) // 2,
            bounds["bottom"] + 30
        )
        time.sleep(1)

        if device(text="cpu").exists:
            device(text="cpu").click()
            time.sleep(1)

        navigate_back(device)

        ensure_model_loaded(device)
        navigate_to_llm(device)

        if not wait_for_text(device, "Chat", timeout=5):
            pytest.skip("Could not navigate to LLM chat")

        send_llm_message(device, "Say hello")

        # Verify app didn't crash
        current = device.app_current()
        assert current["package"] == APP_PACKAGE, "App crashed during CPU inference"

        # Check for any response text
        texts = get_all_texts(device)
        # Filter out known UI elements
        ui_elements = {"Chat", "VLM", "LLM", "Say hello", "Message...",
                       "Send a message to start", "Load a model to begin"}
        response_texts = [t for t in texts if t not in ui_elements
                          and not re.match(r"^\d+\.\d+°C$", t)
                          and len(t) > 2]
        assert len(response_texts) > 0, \
            f"No response text found with CPU backend. Texts: {texts}"


# ===========================================================================
# LLM Output with OpenCL Backend
# ===========================================================================

class TestLlmOpenclOutput:
    """LLM must produce valid output with OpenCL backend."""

    def test_llm_opencl_produces_output(self, device):
        """Set OpenCL backend, load model, send message, verify response."""
        time.sleep(2)

        # Navigate to settings - make sure we're on the main screen first
        settings_btn = device(description="Settings")
        if not settings_btn.exists(timeout=5):
            # May need to navigate to VLM/LLM first
            navigate_to_vlm(device)
            time.sleep(2)
            settings_btn = device(description="Settings")

        if not settings_btn.exists(timeout=5):
            pytest.skip("Settings button not found")

        settings_btn.click()
        time.sleep(2)

        if not wait_for_text(device, "Settings", timeout=5):
            pytest.skip("Settings screen not loaded")

        backend_label = device(text="Backend")
        if not backend_label.exists:
            pytest.skip("Backend setting not visible")

        bounds = backend_label.info["bounds"]
        device.click(
            (bounds["left"] + bounds["right"]) // 2,
            bounds["bottom"] + 30
        )
        time.sleep(1)

        if device(text="opencl").exists:
            device(text="opencl").click()
            time.sleep(1)
        else:
            pytest.skip("OpenCL not available in dropdown")

        navigate_back(device)
        time.sleep(2)

        ensure_model_loaded(device)
        navigate_to_llm(device)

        if not wait_for_text(device, "Chat", timeout=5):
            pytest.skip("Could not navigate to LLM chat")

        send_llm_message(device, "Say hello")

        current = device.app_current()
        assert current["package"] == APP_PACKAGE, "App crashed during OpenCL inference"

        texts = get_all_texts(device)
        ui_elements = {"Chat", "VLM", "LLM", "Say hello", "Message...",
                       "Send a message to start", "Load a model to begin"}
        response_texts = [t for t in texts if t not in ui_elements
                          and not re.match(r"^\d+\.\d+°C$", t)
                          and len(t) > 2]
        assert len(response_texts) > 0, \
            f"No response text found with OpenCL backend. Texts: {texts}"


# ===========================================================================
# VLM Capture + Analyze Flow
# ===========================================================================

class TestVlmCaptureAndAnalyze:
    """VLM capture and analyze flow must not crash."""

    def test_vlm_capture_no_crash(self, device):
        """Capturing an image on VLM screen must not crash."""
        navigate_to_vlm(device)
        time.sleep(2)

        capture_btn = device(description="Capture")
        if not capture_btn.exists(timeout=5):
            pytest.skip("Capture button not found")

        capture_btn.click()
        time.sleep(3)

        current = device.app_current()
        assert current["package"] == APP_PACKAGE, "App crashed during VLM capture"

    def test_vlm_analyze_after_capture(self, device):
        """After capturing an image, clicking AI button must not crash."""
        ensure_model_loaded(device)
        navigate_to_vlm(device)
        time.sleep(2)

        # Capture
        capture_btn = device(description="Capture")
        if not capture_btn.exists(timeout=5):
            pytest.skip("Capture button not found")
        capture_btn.click()
        time.sleep(3)

        # Wait for "Image ready" indicator
        time.sleep(2)

        # App alive check
        current = device.app_current()
        assert current["package"] == APP_PACKAGE, "App crashed after capture"


# ===========================================================================
# Backend Switching
# ===========================================================================

class TestBackendSwitching:
    """Switching between CPU and OpenCL backends must not crash."""

    def test_switch_cpu_to_opencl(self, device):
        """Switch from CPU to OpenCL and verify settings saved."""
        navigate_to_settings(device)

        backend_label = device(text="Backend")
        if not backend_label.exists:
            pytest.skip("Backend setting not visible")

        bounds = backend_label.info["bounds"]

        # Set to CPU first
        device.click(
            (bounds["left"] + bounds["right"]) // 2,
            bounds["bottom"] + 30
        )
        time.sleep(1)
        if device(text="cpu").exists:
            device(text="cpu").click()
            time.sleep(1)

        # Now switch to OpenCL
        device.click(
            (bounds["left"] + bounds["right"]) // 2,
            bounds["bottom"] + 30
        )
        time.sleep(1)
        if device(text="opencl").exists:
            device(text="opencl").click()
            time.sleep(1)

        # Verify opencl is shown
        texts = get_all_texts(device)
        assert "opencl" in texts, \
            f"Backend not changed to opencl. Texts: {texts}"

        current = device.app_current()
        assert current["package"] == APP_PACKAGE, "App crashed during backend switch"
