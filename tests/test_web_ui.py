from pathlib import Path


def test_hold_control_serializes_motion_pulses() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "setInterval(pulse" not in html
    assert "if(pulseLoopRunning)return" in html
    assert "while(holding)" in html
    assert "await api('/api/pulse')" in html


def test_target_controls_support_buttons_typed_phrases_and_speech() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "Blue whale" in html
    assert "Yellow whale" in html
    assert "Apple" in html
    assert "Banana" in html
    assert "choose the yellow" in html
    assert "choose the blue" in html
    assert "choose the apple" in html
    assert "choose the banana" in html
    assert "await api('/api/target',{color:name})" in html
    assert "window.SpeechRecognition||window.webkitSpeechRecognition" in html


def test_camera_refresh_does_not_cancel_an_inflight_image() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "if(refreshInFlight)return" in html
    assert "if(camera.complete)camera.src=" in html
    assert "finally{refreshInFlight=false}" in html
