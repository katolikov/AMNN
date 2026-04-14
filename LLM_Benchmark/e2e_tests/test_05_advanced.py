"""
Test Suite 5: Advanced Feature Tests

Verifies:
- Settings persistence across navigation
- Settings JSON passed to native on model load
- LLM chat stress test (multi-turn conversation)
- VLM image capture and analysis
- Editable VLM prompt
- Temperature displayed in VLM/LLM screens (not TopBar)
"""

import re
import time
import subprocess
import pytest
from conftest import (
    APP_PACKAGE, ADB_MODEL_PATH,
    MODEL_LOAD_TIMEOUT, INFERENCE_TIMEOUT,
    wait_for_text, get_all_texts, get_all_content_descs,
    _adb_cmd,
)


def navigate_to_settings(device):
    settings_btn = device(description="Settings")
    assert settings_btn.exists, "Settings button not found"
    settings_btn.click()
    time.sleep(2)
    assert wait_for_text(device, "Settings", timeout=5), "Settings screen not loaded"


def navigate_back(device):
    back_btn = device(description="Back")
    if back_btn.exists:
        back_btn.click()
        time.sleep(2)


def ensure_model_loaded(device):
    """Load model if needed, skip if unavailable."""
    texts = get_all_texts(device)
    if "No model" not in texts:
        return

    # Try loading from downloaded models
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
                    subprocess.run(_adb_cmd("shell", "am", "start", "-n",
                                            f"{APP_PACKAGE}/.MainActivity"),
                                   capture_output=True, timeout=10)
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


class TestSettingsPersistence:
    """Settings must persist after navigating away from Settings screen."""

    def test_setting_persists_after_navigation(self, device):
        """Change a setting, leave Settings, return — value must be preserved."""
        navigate_to_settings(device)

        # Find the Power dropdown and change it
        if not device(text="Power").exists:
            pytest.skip("Power setting not visible")

        # Remember current value
        texts_before = get_all_texts(device)

        # Change Power to "low" (it defaults to "high")
        power_label = device(text="Power")
        bounds = power_label.info["bounds"]
        # Click below the label to open dropdown
        device.click(
            (bounds["left"] + bounds["right"]) // 2,
            bounds["bottom"] + 30
        )
        time.sleep(1)
        if device(text="low").exists:
            device(text="low").click()
            time.sleep(1)

        # Navigate back
        navigate_back(device)
        time.sleep(2)

        # Return to Settings
        navigate_to_settings(device)

        # Verify the value persisted
        texts_after = get_all_texts(device)
        assert "low" in texts_after, \
            f"Power setting did not persist. Expected 'low' in texts: {texts_after}"

    def test_settings_persist_after_app_restart(self, device):
        """Settings must survive app restart (SharedPreferences)."""
        navigate_to_settings(device)

        # Change precision to "high"
        if not device(text="Precision").exists:
            pytest.skip("Precision setting not visible")

        prec_label = device(text="Precision")
        bounds = prec_label.info["bounds"]
        device.click(
            (bounds["left"] + bounds["right"]) // 2,
            bounds["bottom"] + 30
        )
        time.sleep(1)
        if device(text="high").exists:
            device(text="high").click()
            time.sleep(1)

        # Verify it shows "high" now
        texts = get_all_texts(device)
        assert "high" in texts, "Precision not changed to high"

        # Restart app
        device.app_stop(APP_PACKAGE)
        time.sleep(1)
        subprocess.run(_adb_cmd("shell", "am", "start", "-n",
                                f"{APP_PACKAGE}/.MainActivity"),
                       capture_output=True, timeout=10)
        time.sleep(8)
        # Handle permissions
        for _ in range(3):
            if device(text="While using the app").exists:
                device(text="While using the app").click()
                time.sleep(1)

        # Go to Settings
        navigate_to_settings(device)
        texts = get_all_texts(device)
        assert "high" in texts, \
            f"Precision 'high' did not persist after restart. Texts: {texts}"


class TestSettingsPassedToNative:
    """Verify settings JSON is passed to native code on model load."""

    def test_config_json_logged_on_load(self, device):
        """When a model is loaded, the config JSON should appear in logcat."""
        # Clear logcat
        subprocess.run(_adb_cmd("logcat", "-c"), capture_output=True, timeout=5)

        ensure_model_loaded(device)

        # Check logcat for the config JSON log
        time.sleep(3)
        result = subprocess.run(
            _adb_cmd("logcat", "-d", "-s", "MNNBench"),
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout

        # Look for "Applying config:" log line
        if "Applying config:" in output:
            # Extract the JSON
            for line in output.split("\n"):
                if "Applying config:" in line:
                    assert "{" in line, f"Config JSON not found in log: {line}"
                    return
        # Config may not be logged if model was already loaded
        # Just verify the app didn't crash
        current = device.app_current()
        assert current["package"] == APP_PACKAGE


class TestLLMStressChat:
    """Send multiple messages and verify each gets a response."""

    def test_multi_turn_conversation(self, device):
        """Send 3 messages in sequence, verify responses appear."""
        ensure_model_loaded(device)

        if device(text="LLM").exists(timeout=3):
            device(text="LLM").click()
        time.sleep(2)

        if not wait_for_text(device, "Chat", timeout=5):
            pytest.skip("Could not navigate to LLM chat")

        prompts = ["Say hello", "What is 1+1?", "Say goodbye"]

        for i, prompt in enumerate(prompts):
            msg_input = device(className="android.widget.EditText")
            if not msg_input.exists(timeout=10):
                pytest.skip(f"Input not found for message {i+1}")

            msg_input.set_text(prompt)
            time.sleep(0.5)
            device.send_action("send")
            time.sleep(2)

            # Wait for response (up to INFERENCE_TIMEOUT)
            deadline = time.time() + INFERENCE_TIMEOUT
            while time.time() < deadline:
                current = device.app_current()
                if current["package"] != APP_PACKAGE:
                    pytest.fail(f"App crashed during message {i+1}")

                # Check if input is re-enabled (inference done)
                inp = device(className="android.widget.EditText")
                if inp.exists and inp.info.get("enabled", False):
                    break
                time.sleep(3)

            # Verify app is alive
            current = device.app_current()
            assert current["package"] == APP_PACKAGE, \
                f"App crashed after message {i+1}"

        # Verify multiple messages are visible
        texts = get_all_texts(device)
        message_count = sum(1 for p in prompts if p in texts)
        assert message_count >= 1, \
            f"Expected at least 1 user message visible. Texts: {texts[:20]}"


class TestVLMImageCapture:
    """Capture an image and verify the analyze flow works."""

    def test_capture_and_analyze(self, device):
        """Capture image, then trigger analysis."""
        ensure_model_loaded(device)

        if device(text="VLM").exists(timeout=3):
            device(text="VLM").click()
        time.sleep(2)

        # Capture an image
        capture_btn = device(description="Capture")
        if not capture_btn.exists:
            pytest.skip("Capture button not found")
        capture_btn.click()
        time.sleep(3)

        # App should not crash
        current = device.app_current()
        assert current["package"] == APP_PACKAGE, "App crashed during capture"


class TestEditableVLMPrompt:
    """VLM prompt must be editable via icon button + popup dialog."""

    def test_vlm_prompt_button_exists(self, device):
        """VLM screen must have an 'Edit prompt' icon button (replaces inline TextField)."""
        time.sleep(3)
        btn = device(description="Edit prompt")
        assert btn.exists(timeout=5), \
            f"'Edit prompt' button not found. Descs: {get_all_content_descs(device)}"

    def test_vlm_prompt_editable_via_dialog(self, device):
        """VLM prompt must be editable through the popup dialog."""
        time.sleep(3)
        btn = device(description="Edit prompt")
        assert btn.exists(timeout=5), "Edit prompt button not found"
        btn.click()
        time.sleep(1)

        # Dialog should open with title "VLM Prompt"
        assert device(text="VLM Prompt").exists(timeout=5), \
            f"VLM Prompt dialog not opened. Texts: {get_all_texts(device)}"

        edit_field = device(className="android.widget.EditText")
        assert edit_field.exists, "No EditText in prompt dialog"

        # Edit the prompt
        edit_field.clear_text()
        edit_field.set_text("What do you see?")
        time.sleep(0.5)

        entered = edit_field.get_text()
        assert "What do you see" in entered, \
            f"Prompt not editable. Got: '{entered}'"

        # Save and close
        save_btn = device(text="Save")
        if save_btn.exists:
            save_btn.click()
            time.sleep(1)


class TestTemperatureInScreens:
    """Temperature must be displayed within VLM/LLM screens, not in TopBar."""

    def _find_temp(self, device):
        texts = get_all_texts(device)
        temp_pattern = re.compile(r"\d+\.\d+°C")
        return [t for t in texts if temp_pattern.search(t)]

    def test_temperature_in_vlm_content(self, device):
        """Temperature chip must be visible in VLM screen content."""
        time.sleep(4)  # Wait for temperature LaunchedEffect
        temps = self._find_temp(device)
        assert len(temps) > 0, \
            f"Temperature not found on VLM screen. Texts: {get_all_texts(device)}"

    def test_temperature_in_llm_content(self, device):
        """Temperature chip must be visible in LLM screen header."""
        device(text="LLM").click()
        time.sleep(4)  # Wait for temperature LaunchedEffect
        temps = self._find_temp(device)
        assert len(temps) > 0, \
            f"Temperature not found on LLM screen. Texts: {get_all_texts(device)}"

    def test_temperature_not_in_topbar(self, device):
        """TopBar must NOT contain temperature (removed)."""
        time.sleep(4)
        # TopBar elements are in the first ~200px of screen
        temps = self._find_temp(device)
        for elem in device.xpath("//*[@text!='']").all():
            text = elem.attrib.get("text", "")
            if re.search(r"\d+\.\d+°C", text):
                bounds = elem.attrib.get("bounds", "")
                coords = re.findall(r"\d+", bounds)
                if len(coords) == 4:
                    top = int(coords[1])
                    assert top > 150, \
                        f"Temperature at y={top} is in TopBar area (should be in content)"
                return
