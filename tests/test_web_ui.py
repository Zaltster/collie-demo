from pathlib import Path


def test_one_click_follow_is_owned_by_robot_service() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "setInterval(pulse" not in html
    assert "pulseLoopRunning" not in html
    assert "while(following)" not in html
    assert "await api('/api/pulse')" not in html
    assert "follow.onclick=beginFollow" in html
    assert "await api('/api/follow',{confirmation:'TARGET AND PATH CLEAR'})" in html
    assert "Robot-side follow is active" in html
    assert "PRESS AND HOLD" not in html
    assert "pointerdown" not in html
    assert "addEventListener('blur',endFollow)" not in html
    assert "visibilitychange" not in html
    assert "addEventListener('pagehide',endFollow)" in html
    assert "keepalive:true" in html


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


def test_tracker_confidence_is_not_presented_as_live_yolo_confidence() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "tracker awaiting YOLO check" in html
    assert "observation.confidence===null" in html
    assert "revalidation_failures" in html
    assert "revalidation_failures_required" in html


def test_stage_health_is_visible_in_the_ui() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "STAGE READY" in html
    assert "failedHealth" in html
    assert "gpu_ready" not in html  # rendered generically from the health object
    assert "YOLO verified" in html
    assert "misses ${misses}/${required}" in html
    assert "WAITING FOR STABLE TRACK" in html
    assert "WAITING FOR YOLO" in html
    assert "WAITING FOR FRESH FRAME" in html
    assert "s.follow_readiness" in html
