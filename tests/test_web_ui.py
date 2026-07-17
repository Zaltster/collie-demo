from pathlib import Path


def test_hold_control_serializes_motion_pulses() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "setInterval(pulse" not in html
    assert "if(pulseLoopRunning)return" in html
    assert "while(holding)" in html
    assert "await api('/api/pulse')" in html
