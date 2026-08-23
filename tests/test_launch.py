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


def test_auto_install_dependencies_defaults_to_false(monkeypatch):
    monkeypatch.delenv("DRIFTSYNC_AUTO_INSTALL", raising=False)

    assert launch.auto_install_dependencies() is False


def test_auto_install_dependencies_accepts_truthy_env(monkeypatch):
    monkeypatch.setenv("DRIFTSYNC_AUTO_INSTALL", "1")

    assert launch.auto_install_dependencies() is True
