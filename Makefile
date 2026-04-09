LIB = blue_ml

dev-install:
	uv sync --group dev

test-install:
	uv sync --group test
