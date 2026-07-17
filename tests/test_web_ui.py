from pathlib import Path


def test_one_click_follow_serializes_motion_pulses() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "setInterval(pulse" not in html
    assert "if(pulseLoopRunning)return" in html
    assert "while(following)" in html
    assert "await api('/api/pulse')" in html
    assert "follow.onclick=beginFollow" in html
    assert "await api('/api/arm',{confirmation:'TARGET AND PATH CLEAR'})" in html
    assert "PRESS AND HOLD" not in html
    assert "pointerdown" not in html
    assert "addEventListener('blur',endFollow)" not in html
    assert "visibilitychange" not in html
    assert "addEventListener('pagehide',endFollow)" in html


def test_each_detection_gets_an_exact_select_control() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "for(const [index,item] of state.detections.entries())" in html
    assert "choose.onclick=()=>chooseTarget(item.label,item.center)" in html
    assert "await api('/api/target',{target:name,center})" in html
    assert "closestSelectedIndex" in html
    assert "FOLLOW SELECTED FRUIT" in html
    assert "Blue whale" not in html
    assert "Yellow whale" not in html
    assert "SpeechRecognition" not in html


def test_camera_refresh_does_not_cancel_an_inflight_image() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "if(refreshInFlight)return" in html
    assert "if(camera.complete)camera.src=" in html
    assert "finally{refreshInFlight=false}" in html
