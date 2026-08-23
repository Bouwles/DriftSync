import launch


def test_find_missing_dependencies_reports_import_errors(monkeypatch):
    def fake_import(name):
        if name == "missing_mod":
            raise ImportError("missing")
        return object()

    monkeypatch.setattr(launch.importlib, "import_module", fake_import)

    missing = launch.find_missing_dependencies(
        [("ok_mod", "ok-pkg", "OK package"), ("missing_mod", "missing-pkg", "Missing package")]
    )

    assert missing == [("missing_mod", "missing-pkg", "Missing package")]
