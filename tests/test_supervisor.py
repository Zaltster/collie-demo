from collie_demo.supervisor import process_failure_is_fatal, process_probe_ok


def test_process_probe_accepts_responsive_but_sensor_degraded_app() -> None:
    assert process_probe_ok(
        200,
        {
            "ok": True,
            "stage_ready": False,
            "health": {"produce_live": False},
            "armed": False,
        },
    )


def test_process_probe_rejects_bad_or_malformed_api_response() -> None:
    assert process_probe_ok(503, {"ok": True}) is False
    assert process_probe_ok(200, {"ok": False}) is False
    assert process_probe_ok(200, "not-json-status") is False


def test_supervisor_ignores_cold_cuda_warmup_failures() -> None:
    assert process_failure_is_fatal(
        failures=12,
        failure_limit=12,
        elapsed_s=20.0,
        startup_grace_s=60.0,
    ) is False
    assert process_failure_is_fatal(
        failures=12,
        failure_limit=12,
        elapsed_s=60.0,
        startup_grace_s=60.0,
    ) is True
