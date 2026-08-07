.PHONY: install test check zip

install:
	python -m pip install -e '.[test]'

test:
	pytest

check:
	python -m compileall -q src
	bash -n scripts/*.sh scripts/lib/*.sh

zip:
	cd .. && zip -r gigapose-efficiency-reproduction.zip gigapose-efficiency-reproduction \
		-x '*/__pycache__/*' '*/.pytest_cache/*' '*/.git/*'
