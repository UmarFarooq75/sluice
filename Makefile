# sluice — the protocol as commands, so no rung has to remember it.
#
# The standing protocol (docs/lablog.md) used to live in each experiment's own
# run.sh, and they drifted: two different definitions of "available RAM" were in
# the tree at the same time, one of them using a page size that is wrong on Apple
# Silicon. `make gate` is now the single implementation.
.PHONY: help gate gate-ref static doctor urls build install

help:
	@echo "sluice"
	@echo "  make install    fetch + patch + build llama.cpp, build the driver, make .venv"
	@echo "  make build      rebuild the streaming driver only"
	@echo "  make gate-ref   snapshot the current engine as the byte-identical reference"
	@echo "                  (run BEFORE editing the engine)"
	@echo "  make gate       the standing protocol: static checks + bit-exact gate +"
	@echo "                  byte-identical-off cmp. Model legs self-defer when the"
	@echo "                  machine cannot host them; that is PARTIAL, never green."
	@echo "  make static     static checks only — never touches the model"
	@echo "  make doctor     preflight diagnostics + honest speed prediction"
	@echo "  make urls       HEAD-check every model URL (downloads nothing)"

install:
	bash scripts/install.sh

build:
	bash scripts/build_driver.sh

gate-ref:
	bash scripts/gate.sh --ref

gate:
	bash scripts/gate.sh

static:
	bash scripts/gate.sh --static

doctor:
	@python3 cli/sluice doctor $(MODEL)

urls:
	@python3 cli/sluice urls
