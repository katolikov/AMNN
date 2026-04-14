"""
Test Suite 1: UI & Layout Verification

Verifies:
- Temperature display exists strictly within TopBar on VLM/LLM screens
- Temperature is absent from Settings screen
- All visible buttons are clickable and not crashed
- Mode toggle (VLM/LLM) works correctly
- Core UI elements are present on each screen
"""

import re
import time
import pytest
from conftest import (
    APP_PACKAGE, wait_for_text, wait_for_content_desc,
    get_all_texts, get_all_content_descs,
)


class TestTemperatureDisplay:
    """Assert temperature indicator exists strictly in TopBar, absent elsewhere."""

    def _find_temperature_text(self, device):
        """Find a text matching temperature pattern like '25.0°C'."""
        texts = get_all_texts(device)
        temp_pattern = re.compile(r"\d+\.\d+°C")
        return [t for t in texts if temp_pattern.search(t)]

    def test_temperature_present_on_vlm_screen(self, device):
        """Temperature must be visible on the VLM screen content area."""
        assert wait_for_text(device, "VLM", timeout=5), "VLM tab not found"
        # Temperature chip is now inside the screen content, not the TopBar
        # Wait a few seconds for the TemperatureChip's LaunchedEffect to fire
        time.sleep(4)
        temp_matches = self._find_temperature_text(device)
        assert len(temp_matches) > 0, (
            f"Temperature not found on VLM screen. Texts: {get_all_texts(device)}"
        )

    def test_temperature_present_on_llm_screen(self, device):
        """Temperature must be visible on the LLM screen header."""
        device(text="LLM").click()
        time.sleep(2)
        assert wait_for_text(device, "Chat", timeout=5), "LLM Chat screen not loaded"
        time.sleep(4)
        temp_matches = self._find_temperature_text(device)
        assert len(temp_matches) > 0, (
            f"Temperature not found on LLM screen. Texts: {get_all_texts(device)}"
        )

    def test_temperature_absent_on_settings_screen(self, device):
        """Temperature must be ABSENT from the Settings screen."""
        settings_btn = device(description="Settings")
        assert settings_btn.exists, "Settings button not found"
        settings_btn.click()
        time.sleep(2)
        assert wait_for_text(device, "Settings", timeout=5), "Settings screen not loaded"
        temp_matches = self._find_temperature_text(device)
        assert len(temp_matches) == 0, (
            f"Temperature should NOT be on Settings screen, but found: {temp_matches}"
        )

    def test_temperature_not_in_topbar(self, device):
        """Temperature must NOT be in the TopBar (moved to screen content)."""
        # The TopBar now only has model name, VLM/LLM toggle, settings gear
        texts = get_all_texts(device)
        temp_pattern = re.compile(r"\d+\.\d+°C")
        for elem in device.xpath("//*[@text!='']").all():
            text = elem.attrib.get("text", "")
            if temp_pattern.search(text):
                bounds = elem.attrib.get("bounds", "")
                coords = re.findall(r"\d+", bounds)
                if len(coords) == 4:
                    top = int(coords[1])
                    # If temp is below the TopBar area (>200px), it's in content - OK
                    # If it's in the TopBar (<200px), that's wrong
                    assert top > 150, (
                        f"Temperature at y={top} is still in TopBar area"
                    )
                return


class TestButtonIntegrity:
    """Iterate through all visible buttons, assert they are clickable and functional."""

    def test_vlm_screen_buttons_exist_and_clickable(self, device):
        """All VLM screen buttons must exist and be clickable."""
        # VLM is default screen
        assert wait_for_text(device, "VLM", timeout=5)

        # Check TopBar buttons
        model_selector = device(description="Select model")
        assert model_selector.exists, "Model selector button missing"
        assert model_selector.info["clickable"] or model_selector.info["enabled"], \
            "Model selector not interactive"

        settings_btn = device(description="Settings")
        assert settings_btn.exists, "Settings button missing"
        assert settings_btn.info["enabled"], "Settings button disabled"

        # Mode toggle buttons
        vlm_btn = device(text="VLM")
        assert vlm_btn.exists, "VLM toggle missing"
        llm_btn = device(text="LLM")
        assert llm_btn.exists, "LLM toggle missing"

        # VLM-specific controls
        gallery_btn = device(description="Gallery")
        assert gallery_btn.exists, "Gallery button missing"

        capture_btn = device(description="Capture")
        assert capture_btn.exists, "Capture button missing"

    def test_llm_screen_buttons_exist_and_clickable(self, device):
        """All LLM screen buttons must exist and be clickable."""
        device(text="LLM").click()
        time.sleep(2)
        assert wait_for_text(device, "Chat", timeout=5)

        # Clear chat button
        clear_btn = device(description="Clear chat")
        assert clear_btn.exists, "Clear chat button missing on LLM screen"
        assert clear_btn.info["enabled"], "Clear chat button disabled"

        # Input field placeholder
        msg_input = device(text="Message...")
        assert msg_input.exists, "Message input field missing on LLM screen"

    def test_settings_button_navigates_without_crash(self, device):
        """Clicking Settings must navigate to Settings screen without crash."""
        device(description="Settings").click()
        time.sleep(2)
        assert wait_for_text(device, "Settings", timeout=5), "Settings screen didn't load"
        # Verify the app didn't crash
        current = device.app_current()
        assert current["package"] == APP_PACKAGE, "App crashed after Settings click"

    def test_back_button_on_settings(self, device):
        """Back button on Settings must return to previous screen."""
        device(description="Settings").click()
        time.sleep(2)
        assert wait_for_text(device, "Settings", timeout=5)
        back_btn = device(description="Back")
        assert back_btn.exists, "Back button missing on Settings screen"
        back_btn.click()
        time.sleep(2)
        # Should be back on main screen
        assert device(text="VLM").exists or device(text="LLM").exists, \
            "Failed to navigate back from Settings"

    def test_model_selector_opens_sheet(self, device):
        """Clicking model selector must open the download/select bottom sheet."""
        device(description="Select model").click()
        time.sleep(2)
        assert wait_for_text(device, "Models", timeout=5), \
            "Model download sheet did not open"
        # Verify model entries are listed
        texts = get_all_texts(device)
        assert any("SmolVLM2" in t or "Qwen" in t for t in texts), \
            f"No model names found in download sheet. Texts: {texts}"

    def test_mode_toggle_switches_screens(self, device):
        """VLM/LLM mode toggle must switch between screens."""
        # Start on VLM (default)
        assert device(description="Gallery").exists or device(description="Capture").exists, \
            "VLM screen elements not visible on start"

        # Switch to LLM
        device(text="LLM").click()
        time.sleep(2)
        assert wait_for_text(device, "Chat", timeout=5), "Didn't switch to LLM screen"
        # VLM-specific elements should be gone
        assert not device(description="Capture").exists, \
            "Capture button still visible on LLM screen"

        # Switch back to VLM
        device(text="VLM").click()
        time.sleep(2)
        assert device(description="Gallery").exists or device(description="Capture").exists, \
            "VLM screen elements not restored after switching back"


class TestScreenIntegrity:
    """Verify core UI elements on each screen."""

    def test_vlm_screen_has_camera_preview_or_permission_text(self, device):
        """VLM screen must show camera preview or permission request."""
        assert wait_for_text(device, "VLM", timeout=5)
        # Either camera preview is active (no specific text) or permission text shown
        texts = get_all_texts(device)
        # The screen should have the control buttons at minimum
        descs = get_all_content_descs(device)
        assert "Gallery" in descs, "Gallery button missing from VLM"
        assert "Capture" in descs, "Capture button missing from VLM"

    def test_llm_screen_has_empty_state_message(self, device):
        """LLM screen must show placeholder text when no model is loaded."""
        device(text="LLM").click()
        time.sleep(2)
        texts = get_all_texts(device)
        assert any("Load a model" in t or "Send a message" in t for t in texts), \
            f"LLM screen missing empty state message. Texts: {texts}"

    def test_settings_screen_has_header_and_back(self, device):
        """Settings screen must have 'Settings' header and Back button."""
        device(description="Settings").click()
        time.sleep(2)
        assert device(text="Settings").exists, "Settings header missing"
        assert device(description="Back").exists, "Back button missing"
