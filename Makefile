.PHONY: install install-dev test smoke quick-demo launch

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements-dev.txt

test:
	python -m pytest

smoke:
	python -m driftsync.smoke

quick-demo:
	python run_experiment.py --quick --sessions 5 --trials 80 --epochs 5

launch:
	python launch.py
