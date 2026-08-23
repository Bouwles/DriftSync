from scripts.generate_showcase_assets import H, W, gif_frames, screenshot_overview


def test_showcase_overview_has_expected_size():
    image = screenshot_overview()

    assert image.size == (W, H)


def test_showcase_gif_frames_are_consistent_size():
    frames = gif_frames()

    assert len(frames) == 24
    assert all(frame.size == (W, H) for frame in frames)
