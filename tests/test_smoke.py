from driftsync.smoke import run_smoke_checks


def test_smoke_checks_report_core_contracts():
    checks = run_smoke_checks()

    assert "feature contract: 15 features" in checks
    assert "realtime feature count: aligned" in checks
    assert "lstm input_dim: aligned" in checks
    assert "transformer input_dim: aligned" in checks
