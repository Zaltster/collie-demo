from pathlib import Path


def test_hold_control_serializes_motion_pulses() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "setInterval(pulse" not in html
    assert "if(pulseLoopRunning)return" in html
    assert "while(holding)" in html
    assert "await api('/api/pulse')" in html


def test_target_controls_support_buttons_typed_phrases_and_speech() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "Choose blue" in html
    assert "Choose yellow" in html
    assert "choose the yellow" in html
    assert "choose the blue" in html
    assert "await api('/api/target',{color})" in html
    assert "window.SpeechRecognition||window.webkitSpeechRecognition" in html
