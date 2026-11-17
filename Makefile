PYTHON ?= python3
PORT ?= 8666

.PHONY: setup init-db build start stop status release

setup:
	PYTHON="$(PYTHON)" bash scripts/setup.sh

init-db:
	bash scripts/init_db.sh

build:
	npm --prefix frontend run build

start:
	PORT="$(PORT)" bash start.sh

stop:
	bash stop.sh

status:
	.venv/bin/python scripts/service.py status

release:
	bash scripts/build_release.sh
