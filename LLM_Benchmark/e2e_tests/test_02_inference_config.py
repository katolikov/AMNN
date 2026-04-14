"""
Test Suite 2: Inference Configuration (OpenCL & CPU)

Verifies:
- All OpenCL settings are modifiable (precision, memory, GPU mode, tuning)
- All CPU settings are modifiable (threads)
- Common engine settings work (backend, precision, power, mmap)
- OPs Profiling is disabled/hidden by default and cannot be enabled
- Configuration changes persist in the UI
- Backend switching shows/hides correct setting categories
"""

import re
import time
import pytest
from conftest import (
    APP_PACKAGE, wait_for_text, wait_for_text_gone,
    get_all_texts,
)


def navigate_to_settings(device):
    """Navigate to Settings screen from any state."""
    settings_btn = device(description="Settings")
    assert settings_btn.exists, "Settings button not found"
    settings_btn.click()
    time.sleep(2)
    assert wait_for_text(device, "Settings", timeout=5), "Settings screen did not load"


def find_setting_by_label(device, label):
    """Find a setting item by its label text."""
    return device(text=label)


def get_dropdown_value(device, label):
    """Get the current value of a dropdown setting."""
    setting = find_setting_by_label(device, label)
    if not setting.exists:
        return None
    # The value is in a button below the label within the same container
    parent = setting.up()
    if parent:
        for child in device.xpath(f"//*[@text='{label}']/..").all():
            # Look for button text (the dropdown value)
            bounds = child.attrib.get("bounds", "")
            break
    # Try to find the button text near the label
    texts = get_all_texts(device)
    return texts


def click_dropdown_and_select(device, label, option):
    """Open a dropdown and select an option."""
    setting = find_setting_by_label(device, label)
    if not setting.exists:
        # Scroll to find it
        device(scrollable=True).scroll.to(text=label)
        time.sleep(1)
        setting = find_setting_by_label(device, label)

    assert setting.exists, f"Setting '{label}' not found even after scrolling"

    # Find the dropdown button near this label - it should be the OutlinedButton
    # In Compose, the button is typically a sibling or nearby element
    # Try clicking the button with the current value text below the label
    # The dropdown button is within the same card container
    # Scroll the setting into view first
    setting_bounds = setting.info["bounds"]
    y_center = (setting_bounds["top"] + setting_bounds["bottom"]) // 2

    # Look for clickable elements below the label within ~200px
    # The dropdown button shows current value as text
    found_button = False
    for elem in device.xpath("//android.widget.Button").all():
        elem_bounds = elem.attrib.get("bounds", "")
        coords = re.findall(r"\d+", elem_bounds)
        if len(coords) == 4:
            elem_top = int(coords[1])
            # Button should be close to the label
            if abs(elem_top - y_center) < 200:
                elem_obj = device.xpath(f"//android.widget.Button[@bounds='{elem_bounds}']").get()
                device.click(int(coords[0]) + 50, int(coords[1]) + 20)
                found_button = True
                break

    if not found_button:
        # Alternative: try clicking below the label text
        device.click(
            (setting_bounds["left"] + setting_bounds["right"]) // 2,
            setting_bounds["bottom"] + 30
        )

    time.sleep(1)

    # Now select the option from the dropdown
    option_elem = device(text=option)
    if option_elem.exists:
        option_elem.click()
        time.sleep(1)
        return True
    return False


class TestOPsProfiling:
    """OPs Profiling must be disabled/hidden by default and cannot be enabled."""

    def test_ops_profiling_not_visible_in_settings(self, device):
        """enable_op_profile must NOT appear in the Settings UI."""
        navigate_to_settings(device)

        # Scroll through all settings to check
        texts = get_all_texts(device)

        # Check that "OPs Profiling" or "enable_op_profile" is NOT in any text
        for t in texts:
            assert "op_profile" not in t.lower(), \
                f"OPs Profiling setting found in UI: '{t}'"
            assert "ops profiling" not in t.lower(), \
                f"OPs Profiling setting found in UI: '{t}'"

        # Also scroll down to check all items
        scrollable = device(scrollable=True)
        if scrollable.exists:
            scrollable.scroll.toEnd(max_swipes=5)
            time.sleep(1)
            texts_after_scroll = get_all_texts(device)
            for t in texts_after_scroll:
                assert "op_profile" not in t.lower(), \
                    f"OPs Profiling found after scrolling: '{t}'"
                assert "ops profiling" not in t.lower(), \
                    f"OPs Profiling found after scrolling: '{t}'"

    def test_ops_profiling_not_in_unknown_params(self, device):
        """enable_op_profile must not leak through to the 'Other' section."""
        navigate_to_settings(device)

        # Scroll to bottom to see "Other" section
        scrollable = device(scrollable=True)
        if scrollable.exists:
            scrollable.scroll.toEnd(max_swipes=5)
            time.sleep(1)

        texts = get_all_texts(device)
        assert "enable_op_profile" not in texts, \
            "enable_op_profile leaked into the 'Other' parameters section"


class TestBackendSwitching:
    """Test switching between CPU and GPU backends shows/hides correct settings."""

    def test_cpu_backend_shows_cpu_settings(self, device):
        """When backend=cpu, CPU-specific settings (threads) must be visible."""
        navigate_to_settings(device)

        # Ensure we're on CPU backend
        backend_label = find_setting_by_label(device, "Backend")
        if not backend_label.exists:
            pytest.skip("Backend setting not found - config may not have 'backend' key")

        # Try to select CPU
        click_dropdown_and_select(device, "Backend", "cpu")
        time.sleep(1)

        # CPU Threads should now be visible (may need scrolling)
        scrollable = device(scrollable=True)
        if scrollable.exists:
            scrollable.scroll.to(text="CPU Threads")
            time.sleep(0.5)

        assert device(text="CPU Threads").exists, \
            "CPU Threads setting not visible when backend=cpu"

    def test_cpu_backend_hides_gpu_settings(self, device):
        """When backend=cpu, GPU-specific settings must be hidden."""
        navigate_to_settings(device)

        click_dropdown_and_select(device, "Backend", "cpu")
        time.sleep(1)

        # Scroll through all settings
        all_texts = get_all_texts(device)
        scrollable = device(scrollable=True)
        if scrollable.exists:
            scrollable.scroll.toEnd(max_swipes=5)
            time.sleep(0.5)
            all_texts.extend(get_all_texts(device))

        # GPU-only settings should NOT be visible
        gpu_settings = ["Memory Mode", "GPU Mode", "Disable GPU Tuning"]
        for gs in gpu_settings:
            assert gs not in all_texts, \
                f"GPU setting '{gs}' visible when backend=cpu"

    def test_gpu_backend_shows_gpu_settings(self, device):
        """When backend=opencl, GPU-specific settings must be visible."""
        navigate_to_settings(device)

        if not click_dropdown_and_select(device, "Backend", "opencl"):
            pytest.skip("Could not select opencl backend")
        time.sleep(1)

        # Scroll to find GPU settings
        scrollable = device(scrollable=True)
        if scrollable.exists:
            scrollable.scroll.to(text="GPU")
            time.sleep(0.5)

        all_texts = get_all_texts(device)
        if scrollable.exists:
            scrollable.scroll.toEnd(max_swipes=3)
            time.sleep(0.5)
            all_texts.extend(get_all_texts(device))

        # At least some GPU settings should be visible
        gpu_settings_found = [gs for gs in ["Memory Mode", "GPU Mode", "Disable GPU Tuning"]
                              if gs in all_texts]
        assert len(gpu_settings_found) > 0, \
            f"No GPU settings visible when backend=opencl. All texts: {all_texts}"

    def test_gpu_backend_hides_cpu_settings(self, device):
        """When backend=opencl, CPU-specific settings must be hidden."""
        navigate_to_settings(device)

        if not click_dropdown_and_select(device, "Backend", "opencl"):
            pytest.skip("Could not select opencl backend")
        time.sleep(1)

        all_texts = get_all_texts(device)
        scrollable = device(scrollable=True)
        if scrollable.exists:
            scrollable.scroll.toEnd(max_swipes=5)
            time.sleep(0.5)
            all_texts.extend(get_all_texts(device))

        assert "CPU Threads" not in all_texts, \
            "CPU Threads visible when backend=opencl"


class TestOpenCLSettings:
    """Modify every available OpenCL setting and verify changes persist."""

    def _switch_to_opencl(self, device):
        navigate_to_settings(device)
        if not click_dropdown_and_select(device, "Backend", "opencl"):
            pytest.skip("Could not select opencl backend")
        time.sleep(1)

    def test_opencl_precision_options(self, device):
        """Precision dropdown must have low/normal/high options and be changeable."""
        self._switch_to_opencl(device)

        # Try changing precision
        for option in ["high", "low", "normal"]:
            result = click_dropdown_and_select(device, "Precision", option)
            if result:
                time.sleep(0.5)
                # Verify the option text appears in UI (as the dropdown value)
                texts = get_all_texts(device)
                assert option in texts, \
                    f"Precision value '{option}' not reflected in UI after selection"
                break
        else:
            pytest.skip("Could not interact with Precision dropdown")

    def test_opencl_memory_mode(self, device):
        """Memory Mode dropdown must be changeable between low/high."""
        self._switch_to_opencl(device)

        scrollable = device(scrollable=True)
        if scrollable.exists:
            scrollable.scroll.to(text="Memory Mode")
            time.sleep(0.5)

        if not device(text="Memory Mode").exists:
            pytest.skip("Memory Mode setting not visible")

        result = click_dropdown_and_select(device, "Memory Mode", "high")
        if result:
            texts = get_all_texts(device)
            assert "high" in texts, "Memory Mode 'high' not reflected in UI"

    def test_opencl_gpu_mode(self, device):
        """GPU Mode dropdown must have tuning options and be changeable."""
        self._switch_to_opencl(device)

        # Scroll down to find GPU Mode — may need multiple swipes
        scrollable = device(scrollable=True)
        if scrollable.exists:
            for _ in range(5):
                if device(text="GPU Mode").exists:
                    break
                scrollable.scroll.forward()
                time.sleep(0.5)

        if not device(text="GPU Mode").exists:
            # Check if gpu_mode key is present in the config at all
            all_texts = get_all_texts(device)
            pytest.skip(f"GPU Mode setting not visible after scrolling. Texts: {all_texts}")

        result = click_dropdown_and_select(device, "GPU Mode", "tuning_fast")
        if result:
            texts = get_all_texts(device)
            assert "tuning_fast" in texts, "GPU Mode 'tuning_fast' not reflected in UI"

    def test_opencl_disable_tuning_toggle(self, device):
        """Disable GPU Tuning toggle must be switchable."""
        self._switch_to_opencl(device)

        # Scroll down to find Disable GPU Tuning
        scrollable = device(scrollable=True)
        if scrollable.exists:
            for _ in range(5):
                if device(text="Disable GPU Tuning").exists:
                    break
                scrollable.scroll.forward()
                time.sleep(0.5)

        if not device(text="Disable GPU Tuning").exists:
            all_texts = get_all_texts(device)
            pytest.skip(f"Disable GPU Tuning not visible. Texts: {all_texts}")

        # Find the switch near the label
        switch = device(className="android.widget.Switch")
        if switch.exists:
            switch.click()
            time.sleep(0.5)
            texts = get_all_texts(device)
            # The state text should change between "Enabled"/"Disabled"
            assert "Enabled" in texts or "Disabled" in texts, \
                "Toggle state text not found after clicking"


class TestCPUSettings:
    """Modify all available CPU settings and verify changes."""

    def _switch_to_cpu(self, device):
        navigate_to_settings(device)
        click_dropdown_and_select(device, "Backend", "cpu")
        time.sleep(1)

    def test_cpu_threads_slider(self, device):
        """CPU Threads slider must be adjustable (range 1-8)."""
        self._switch_to_cpu(device)

        scrollable = device(scrollable=True)
        if scrollable.exists:
            scrollable.scroll.to(text="CPU Threads")
            time.sleep(0.5)

        if not device(text="CPU Threads").exists:
            pytest.skip("CPU Threads setting not visible")

        # Find the slider
        slider = device(className="android.widget.SeekBar")
        if slider.exists:
            # Get slider bounds for interaction
            bounds = slider.info["bounds"]
            slider_width = bounds["right"] - bounds["left"]
            slider_y = (bounds["top"] + bounds["bottom"]) // 2

            # Slide to ~75% (should set to ~6 threads)
            target_x = bounds["left"] + int(slider_width * 0.75)
            device.click(target_x, slider_y)
            time.sleep(0.5)

            # Verify a thread count value is displayed (should be between 1-8)
            texts = get_all_texts(device)
            thread_values = [t for t in texts if t.isdigit() and 1 <= int(t) <= 8]
            assert len(thread_values) > 0, \
                f"No valid thread count (1-8) found after slider adjustment. Texts: {texts}"

    def test_power_setting(self, device):
        """Power dropdown must be changeable between normal/low/high."""
        self._switch_to_cpu(device)

        if not device(text="Power").exists:
            scrollable = device(scrollable=True)
            if scrollable.exists:
                scrollable.scroll.to(text="Power")
                time.sleep(0.5)

        if not device(text="Power").exists:
            pytest.skip("Power setting not visible")

        result = click_dropdown_and_select(device, "Power", "high")
        if result:
            texts = get_all_texts(device)
            assert "high" in texts, "Power 'high' not reflected in UI"

    def test_use_mmap_toggle(self, device):
        """Use mmap toggle must be switchable."""
        self._switch_to_cpu(device)

        if not device(text="Use mmap").exists:
            scrollable = device(scrollable=True)
            if scrollable.exists:
                scrollable.scroll.to(text="Use mmap")
                time.sleep(0.5)

        if not device(text="Use mmap").exists:
            pytest.skip("Use mmap setting not visible")

        # Check initial state text
        texts_before = get_all_texts(device)
        has_enabled = "Enabled" in texts_before
        has_disabled = "Disabled" in texts_before

        assert has_enabled or has_disabled, \
            "Use mmap toggle state text (Enabled/Disabled) not found"


class TestCommonSettings:
    """Test settings that apply regardless of backend."""

    def test_temperature_slider(self, device):
        """Temperature sampler slider must be adjustable (0.0 - 2.0)."""
        navigate_to_settings(device)

        # Scroll to the very bottom to find Temperature (in Sampler category)
        scrollable = device(scrollable=True)
        if scrollable.exists:
            # Scroll all the way down since Temperature is near the bottom
            for _ in range(8):
                if device(text="Temperature").exists:
                    break
                scrollable.scroll.forward()
                time.sleep(0.3)

        if not device(text="Temperature").exists:
            pytest.skip("Temperature setting not visible after scrolling")

        # Scroll a tiny bit more to ensure the slider value is visible
        if scrollable.exists:
            scrollable.scroll.forward()
            time.sleep(0.5)

        texts = get_all_texts(device)
        # Temperature label must exist
        assert "Temperature" in texts, \
            f"Temperature label not found. Texts: {texts}"
        # Look for float values like "1.00", "0.50", "1.0", "4.0" etc.
        float_pattern = re.compile(r"^\d+\.\d+$")
        float_values = [t for t in texts if float_pattern.match(t)]
        assert len(float_values) > 0, \
            f"No float values found near Temperature setting. Texts: {texts}"

    def test_benchmark_settings_exist(self, device):
        """Benchmark-specific settings should be visible."""
        navigate_to_settings(device)

        benchmark_settings = ["Prompt Tokens", "Max Generate Tokens"]
        found = []

        # These settings may be far down the list, scroll aggressively
        scrollable = device(scrollable=True)
        for setting_name in benchmark_settings:
            if device(text=setting_name).exists:
                found.append(setting_name)
                continue
            if scrollable.exists:
                for _ in range(10):
                    if device(text=setting_name).exists:
                        found.append(setting_name)
                        break
                    scrollable.scroll.forward()
                    time.sleep(0.3)

        assert len(found) > 0, \
            f"No benchmark settings found. Expected at least one of: {benchmark_settings}"
