.DEFAULT_GOAL := help
UV ?= uv
PROJECT_ROOT ?= $(CURDIR)
# POSIX-shell quote one argument, including embedded single quotes.
quote = '$(subst ','"'"',$(1))'
CLI = $(call quote,$(UV)) run researchflow
ROOT_ARG = --root $(call quote,$(PROJECT_ROOT))

.PHONY: help doctor smoke experiment analyze view review artifacts freeze-study paper paper-check release
help:
	@printf '%s\n' 'make TARGET [PROJECT_ROOT=path] [UV=uv]' \
	  'setup | data | cold-start [DATA=path] [NO_TEX=1]' \
	  'doctor | smoke' \
	  'experiment CONFIG=path [RUN_ID=id] [FREEZE=path]' \
	  'analyze CONFIG=path' \
	  'view | review CONFIG=path OUTPUT=path' \
	  'artifacts | paper PUBLICATION=path' \
	  'freeze-study CONFIG=path OUTPUT=path' \
	  'paper-check COMPILED=path' \
	  'release SNAPSHOT=path COMPILED=path OUTPUT=path VISUAL_REVIEW=path'

# Missing required variables must fail before invoking a service.
define require
	@test -n $(call quote,$($(1))) || { printf '%s\n' 'Missing $(1); run make help for usage.' >&2; exit 2; }
endef

doctor:
	$(CLI) doctor $(ROOT_ARG)
smoke:
	$(CLI) smoke $(ROOT_ARG)
experiment:
	$(call require,CONFIG)
	$(CLI) experiment $(ROOT_ARG) --config $(call quote,$(CONFIG)) $(if $(strip $(RUN_ID)),--run-id $(call quote,$(RUN_ID))) $(if $(strip $(FREEZE)),--freeze $(call quote,$(FREEZE)))
analyze:
	$(call require,CONFIG)
	$(CLI) analyze $(ROOT_ARG) --config $(call quote,$(CONFIG))
view:
	$(call require,CONFIG)
	$(call require,OUTPUT)
	$(CLI) view $(ROOT_ARG) --config $(call quote,$(CONFIG)) --output $(call quote,$(OUTPUT))
review:
	$(call require,CONFIG)
	$(call require,OUTPUT)
	$(CLI) review $(ROOT_ARG) --config $(call quote,$(CONFIG)) --output $(call quote,$(OUTPUT))
artifacts:
	$(call require,PUBLICATION)
	$(CLI) artifacts $(ROOT_ARG) --publication $(call quote,$(PUBLICATION))
freeze-study:
	$(call require,CONFIG)
	$(call require,OUTPUT)
	$(CLI) freeze-study $(ROOT_ARG) --config $(call quote,$(CONFIG)) --output $(call quote,$(OUTPUT))
paper:
	$(call require,PUBLICATION)
	$(CLI) paper $(ROOT_ARG) --publication $(call quote,$(PUBLICATION))
paper-check:
	$(call require,COMPILED)
	$(CLI) paper-check $(ROOT_ARG) --compiled $(call quote,$(COMPILED))
release:
	$(call require,SNAPSHOT)
	$(call require,COMPILED)
	$(call require,OUTPUT)
	$(call require,VISUAL_REVIEW)
	$(CLI) release $(ROOT_ARG) --snapshot $(call quote,$(SNAPSHOT)) --compiled $(call quote,$(COMPILED)) --output $(call quote,$(OUTPUT)) --visual-review $(call quote,$(VISUAL_REVIEW))

# Repository bootstrap. Existing paper/artifact targets remain computation-free.
PYTHON ?= python3
.PHONY: setup data cold-start
setup:
	$(call quote,$(PYTHON)) tools/cold_start.py setup --root $(call quote,$(PROJECT_ROOT)) --uv $(call quote,$(UV))
data:
	$(call quote,$(PYTHON)) tools/cold_start.py data --root $(call quote,$(PROJECT_ROOT)) --uv $(call quote,$(UV)) $(if $(strip $(DATA)),--existing-data $(call quote,$(DATA)))
cold-start:
	$(call quote,$(PYTHON)) tools/cold_start.py baseline --root $(call quote,$(PROJECT_ROOT)) --uv $(call quote,$(UV)) $(if $(strip $(DATA)),--existing-data $(call quote,$(DATA))) $(if $(filter 1,$(NO_TEX)),--no-tex)
