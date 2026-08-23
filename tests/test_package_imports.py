def test_public_packages_import_cleanly():
    import driftsync
    import driftsync.configs
    import driftsync.data
    import driftsync.evaluation
    import driftsync.ml
    import driftsync.models
    import driftsync.realtime
    import driftsync.simulator
    import driftsync.training
    import driftsync.utils

    assert driftsync is not None
