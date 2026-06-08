.PHONY: install lint fmt typecheck test bench bench-fast pdf clean
install: ; uv sync --extra dev
lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests
typecheck: ; uv run mypy src
test: ; uv run pytest -v
bench: ; uv run ppa bench --out-dir runs/latest
bench-fast: ; uv run ppa bench --out-dir runs/fast --n-docs 6000 --n-queries 250
pdf:
	cd docs/_report && pandoc research_report.md -o ../research_report.pdf \
	    --pdf-engine=typst --toc --toc-depth=2 --number-sections --resource-path=.:../..
clean: ; rm -rf runs/* .pytest_cache .mypy_cache .ruff_cache
