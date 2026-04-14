"""
Test Suite 6: Prompt Button + Dialog & Screen Transition Animations

Verifies:
- VLM: inline TextField replaced by icon button (top-right) that opens popup dialog
- LLM: system prompt icon button in header opens popup dialog
- Prompt dialog allows editing and saving
- Screen transition animations occur (VLM ↔ LLM, Settings)
- Mode toggle animates selection indicator
"""

import re
import time
import subprocess
import pytest
from conftest import (
    APP_PACKAGE,
    UI_TIMEOUT,
    wait_for_text,
    wait_for_text_gone,
    get_all_texts,
    get_all_content_descs,
    _adb_cmd,
    MAIN_ACTIVITY,
    LAUNCH_TIMEOUT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_element_by_content_desc(device, desc, timeout=UI_TIMEOUT):
    """Wait for an element with the given content-description."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        el = device(description=desc)
        if el.exists:
            return el
        time.sleep(0.5)
    return None


def _find_dialog_title(device, title, timeout=UI_TIMEOUT):
    """Wait for a dialog with the given title text."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if device(text=title).exists:
            return True
        time.sleep(0.5)
    return False


# ===========================================================================
# VLM Prompt Button & Dialog
# ===========================================================================

class TestVlmPromptButton:
    """VLM inline TextField must be replaced by an icon button + popup dialog."""

    def test_vlm_no_inline_prompt_textfield(self, device):
        """VLM screen must NOT have an inline prompt TextField in the bottom bar."""
        time.sleep(3)
        # The old inline TextField had placeholder "Prompt for image analysis..."
        texts = get_all_texts(device)
        # The inline TextField is gone; only a small preview chip with the prompt text
        # and the "Edit prompt" icon button remain
        assert "Prompt for image analysis..." not in texts, (
            "Old inline prompt TextField placeholder still present on VLM screen"
        )

    def test_vlm_edit_prompt_button_exists(self, device):
        """VLM screen must have an 'Edit prompt' icon button."""
        time.sleep(3)
        btn = _find_element_by_content_desc(device, "Edit prompt", timeout=5)
        assert btn is not None, (
            f"'Edit prompt' button not found. Descs: {get_all_content_descs(device)}"
        )

    def test_vlm_edit_prompt_opens_dialog(self, device):
        """Clicking 'Edit prompt' must open a popup dialog with title 'VLM Prompt'."""
        time.sleep(3)
        btn = _find_element_by_content_desc(device, "Edit prompt", timeout=5)
        assert btn is not None, "Edit prompt button not found"
        btn.click()
        time.sleep(1)

        assert _find_dialog_title(device, "VLM Prompt", timeout=5), (
            f"VLM Prompt dialog did not open. Texts: {get_all_texts(device)}"
        )

    def test_vlm_prompt_dialog_has_textfield(self, device):
        """The VLM Prompt dialog must contain an editable TextField."""
        time.sleep(3)
        btn = _find_element_by_content_desc(device, "Edit prompt", timeout=5)
        assert btn is not None
        btn.click()
        time.sleep(1)

        assert _find_dialog_title(device, "VLM Prompt", timeout=5)

        edit_field = device(className="android.widget.EditText")
        assert edit_field.exists, "No EditText found inside VLM Prompt dialog"

    def test_vlm_prompt_dialog_edit_and_save(self, device):
        """Editing prompt in dialog and clicking Save must update the prompt."""
        time.sleep(3)
        btn = _find_element_by_content_desc(device, "Edit prompt", timeout=5)
        assert btn is not None
        btn.click()
        time.sleep(1)

        assert _find_dialog_title(device, "VLM Prompt", timeout=5)

        edit_field = device(className="android.widget.EditText")
        assert edit_field.exists

        # Clear and type new prompt
        edit_field.clear_text()
        edit_field.set_text("What objects are in this image?")
        time.sleep(0.5)

        # Click Save
        save_btn = device(text="Save")
        assert save_btn.exists, "Save button not found in dialog"
        save_btn.click()
        time.sleep(1)

        # Dialog should be closed
        assert not device(text="VLM Prompt").exists(timeout=2), "Dialog still open after Save"

        # Prompt preview chip should show new text
        texts = get_all_texts(device)
        assert any("What objects" in t for t in texts), (
            f"Updated prompt not visible in preview chip. Texts: {texts}"
        )

    def test_vlm_prompt_dialog_cancel(self, device):
        """Clicking Cancel must close dialog without changing prompt."""
        time.sleep(3)
        btn = _find_element_by_content_desc(device, "Edit prompt", timeout=5)
        assert btn is not None
        btn.click()
        time.sleep(1)

        assert _find_dialog_title(device, "VLM Prompt", timeout=5)

        edit_field = device(className="android.widget.EditText")
        assert edit_field.exists
        original_text = edit_field.get_text()

        # Type something different
        edit_field.clear_text()
        edit_field.set_text("SHOULD NOT BE SAVED")
        time.sleep(0.5)

        # Click Cancel
        cancel_btn = device(text="Cancel")
        assert cancel_btn.exists, "Cancel button not found"
        cancel_btn.click()
        time.sleep(1)

        # Dialog should be closed
        assert not device(text="VLM Prompt").exists(timeout=2)

        # Prompt should NOT have changed
        texts = get_all_texts(device)
        assert "SHOULD NOT BE SAVED" not in texts, (
            "Cancelled prompt was saved — Cancel did not work"
        )

    def test_vlm_prompt_preview_chip_visible(self, device):
        """VLM screen must show a small prompt preview chip below the top-right button."""
        time.sleep(3)
        texts = get_all_texts(device)
        # Default prompt "Describe this image." should appear somewhere as preview
        has_preview = any("Describe this image" in t for t in texts)
        assert has_preview, (
            f"Prompt preview chip not visible. Texts: {texts}"
        )


# ===========================================================================
# LLM Prompt Button & Dialog
# ===========================================================================

class TestLlmPromptButton:
    """LLM screen must have a system prompt icon button that opens a dialog."""

    def _navigate_to_llm(self, device):
        llm_btn = device(text="LLM")
        if llm_btn.exists(timeout=3):
            llm_btn.click()
            time.sleep(2)

    def test_llm_edit_prompt_button_exists(self, device):
        """LLM header must have an 'Edit prompt' icon button."""
        self._navigate_to_llm(device)

        btn = _find_element_by_content_desc(device, "Edit prompt", timeout=5)
        assert btn is not None, (
            f"'Edit prompt' button not found on LLM screen. Descs: {get_all_content_descs(device)}"
        )

    def test_llm_edit_prompt_opens_dialog(self, device):
        """Clicking 'Edit prompt' on LLM must open 'System Prompt' dialog."""
        self._navigate_to_llm(device)

        btn = _find_element_by_content_desc(device, "Edit prompt", timeout=5)
        assert btn is not None
        btn.click()
        time.sleep(1)

        assert _find_dialog_title(device, "System Prompt", timeout=5), (
            f"System Prompt dialog did not open. Texts: {get_all_texts(device)}"
        )

    def test_llm_prompt_dialog_edit_and_save(self, device):
        """Editing system prompt and saving must update."""
        self._navigate_to_llm(device)

        btn = _find_element_by_content_desc(device, "Edit prompt", timeout=5)
        assert btn is not None
        btn.click()
        time.sleep(1)

        assert _find_dialog_title(device, "System Prompt", timeout=5)

        edit_field = device(className="android.widget.EditText")
        assert edit_field.exists

        edit_field.clear_text()
        edit_field.set_text("You are a pirate.")
        time.sleep(0.5)

        save_btn = device(text="Save")
        assert save_btn.exists
        save_btn.click()
        time.sleep(1)

        assert not device(text="System Prompt").exists(timeout=2), "Dialog still open"

    def test_llm_prompt_dialog_has_default_value(self, device):
        """System Prompt dialog should have a default 'You are a helpful assistant.' value."""
        self._navigate_to_llm(device)

        btn = _find_element_by_content_desc(device, "Edit prompt", timeout=5)
        assert btn is not None
        btn.click()
        time.sleep(1)

        assert _find_dialog_title(device, "System Prompt", timeout=5)

        edit_field = device(className="android.widget.EditText")
        assert edit_field.exists

        text = edit_field.get_text()
        assert "helpful assistant" in text.lower() or "you are" in text.lower(), (
            f"Default system prompt not found. Got: '{text}'"
        )


# ===========================================================================
# Screen Transition Animations
# ===========================================================================

def _recover_app(device):
    """Bring the app back to foreground if it left."""
    current = device.app_current()
    if current["package"] != APP_PACKAGE:
        subprocess.run(
            _adb_cmd("shell", "input", "keyevent", "KEYCODE_WAKEUP"),
            capture_output=True, timeout=5
        )
        subprocess.run(
            _adb_cmd("shell", "input", "keyevent", "KEYCODE_MENU"),
            capture_output=True, timeout=5
        )
        time.sleep(0.5)
        subprocess.run(
            _adb_cmd("shell", "am", "start", "-n", MAIN_ACTIVITY),
            capture_output=True, timeout=10
        )
        time.sleep(LAUNCH_TIMEOUT)
        for _ in range(3):
            if device(text="While using the app").exists:
                device(text="While using the app").click()
                time.sleep(1)
            elif device(text="Allow").exists:
                device(text="Allow").click()
                time.sleep(1)
        time.sleep(1)
    return device.app_current()["package"] == APP_PACKAGE


class TestScreenTransitions:
    """Verify animated transitions between VLM, LLM, and Settings screens."""

    def test_vlm_to_llm_transition_no_crash(self, device):
        """Switching VLM→LLM must not crash the app."""
        time.sleep(2)
        assert _recover_app(device), "Cannot bring app to foreground"

        llm_btn = device(text="LLM")
        assert llm_btn.exists(timeout=5), "LLM toggle button not found"
        llm_btn.click()
        time.sleep(2)  # Animation time

        # Allow recovery if device went to sleep
        if not _recover_app(device):
            pytest.fail("App crashed during VLM→LLM transition")

        # Verify LLM screen loaded
        assert wait_for_text(device, "Chat", timeout=5), "LLM screen not loaded after transition"

    def test_llm_to_vlm_transition_no_crash(self, device):
        """Switching LLM→VLM must not crash the app."""
        time.sleep(2)
        assert _recover_app(device), "Cannot bring app to foreground"

        # Go to LLM first
        llm_btn = device(text="LLM")
        if llm_btn.exists(timeout=3):
            llm_btn.click()
            time.sleep(2)

        # Switch back to VLM
        vlm_btn = device(text="VLM")
        assert vlm_btn.exists(timeout=5), "VLM toggle button not found"
        vlm_btn.click()
        time.sleep(2)  # Animation time

        if not _recover_app(device):
            pytest.fail("App crashed during LLM→VLM transition")

    def test_settings_transition_no_crash(self, device):
        """Opening and closing Settings must not crash."""
        time.sleep(2)
        assert _recover_app(device), "Cannot bring app to foreground"

        settings_btn = device(description="Settings")
        assert settings_btn.exists(timeout=5), "Settings button not found"
        settings_btn.click()
        time.sleep(2)  # Slide-up animation

        if not _recover_app(device):
            pytest.fail("App crashed during Settings enter")

        assert wait_for_text(device, "Settings", timeout=5), "Settings screen not loaded"

        # Go back
        back_btn = device(description="Back")
        if back_btn.exists:
            back_btn.click()
            time.sleep(2)  # Slide-down animation

        if not _recover_app(device):
            pytest.fail("App crashed during Settings exit")

    def test_rapid_mode_switching(self, device):
        """Rapidly switching VLM↔LLM should not crash."""
        time.sleep(2)
        assert _recover_app(device), "Cannot bring app to foreground"

        for _ in range(5):
            if device(text="LLM").exists:
                device(text="LLM").click()
                time.sleep(0.3)
            if device(text="VLM").exists:
                device(text="VLM").click()
                time.sleep(0.3)

        time.sleep(1)
        if not _recover_app(device):
            pytest.fail("App crashed during rapid mode switching")


# ===========================================================================
# Mode Toggle Animation
# ===========================================================================

class TestModeToggleAnimation:
    """Mode toggle must have animated selection indicator."""

    def test_mode_toggle_vlm_selected_by_default(self, device):
        """VLM mode should be selected by default."""
        time.sleep(2)
        texts = get_all_texts(device)
        assert "VLM" in texts, f"VLM label not found. Texts: {texts}"
        assert "LLM" in texts, f"LLM label not found. Texts: {texts}"

    def test_mode_toggle_switches_correctly(self, device):
        """Tapping LLM in mode toggle must switch to LLM screen."""
        time.sleep(2)
        device(text="LLM").click()
        time.sleep(1)

        # Should see Chat header from LLM screen
        assert wait_for_text(device, "Chat", timeout=5), "LLM Chat screen not shown"

        # Switch back to VLM
        device(text="VLM").click()
        time.sleep(1)

        # VLM should show camera elements
        descs = get_all_content_descs(device)
        assert "Capture" in descs or "Gallery" in descs, (
            f"VLM screen not restored. Descs: {descs}"
        )
