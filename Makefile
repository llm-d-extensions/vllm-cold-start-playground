# llm-d cold start: measurement tooling.
#
# The two things you need to know:
#   make selftest                 validate the whole pipeline locally (no GPU)
#   make run NS=<ns> MODEL=<id>   measure a real cold start in a cluster
#
NS ?=
MODEL ?=
POD ?= vllm-coldstart
REPEAT ?= 1
RUN_ID ?= $(shell date -u +%Y%m%d-%H%M%S)
RUNS_DIR ?= runs
VLLM_ARGS ?=
KUBECTL := kubectl $(if $(NS),-n $(NS),)

.PHONY: help selftest configmap deploy run exec fetch report compare clean lint

help:
	@sed -n '3,5p' Makefile
	@echo
	@grep -E '^[a-z-]+:.*?## .*$$' Makefile | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

selftest: ## end-to-end check of probe + poller + report in a container (no GPU)
	tests/selftest.sh

configmap: ## regenerate manifests/generated/probe-configmap.yaml from coldstart/
	scripts/build-probe-configmap.sh $(if $(NS),-n $(NS),)

deploy: configmap ## apply the probe ConfigMap, cache PVC and experiment pod
	$(KUBECTL) apply -f manifests/generated/probe-configmap.yaml
	$(KUBECTL) apply -f manifests/cache-pvc.yaml
	$(KUBECTL) apply -f manifests/pod-exec.yaml

run: ## full experiment: deploy, measure, fetch, report (NS=, MODEL=, REPEAT=)
	@test -n "$(MODEL)" || { echo "usage: make run NS=<ns> MODEL=<hf-id> [REPEAT=3]"; exit 2; }
	scripts/run-experiment.sh $(if $(NS),-n $(NS),) --pod $(POD) \
	  --run-id $(RUN_ID) --repeat $(REPEAT) --model $(MODEL) \
	  --out $(RUNS_DIR) $(if $(VLLM_ARGS),-- $(VLLM_ARGS),)

exec: ## measure again inside an already-running pod (no re-deploy)
	@test -n "$(MODEL)" || { echo "usage: make exec NS=<ns> MODEL=<hf-id>"; exit 2; }
	$(KUBECTL) exec $(POD) -c vllm -- bash /opt/coldstart-src/coldstart-run.sh \
	  --run-id $(RUN_ID) --repeat $(REPEAT) -- --model $(MODEL) $(VLLM_ARGS)

fetch: ## pull run dirs out of the pod and render reports (RUN_ID= or ALL=1)
	scripts/fetch-trace.sh $(if $(NS),-n $(NS),) --pod $(POD) \
	  $(if $(ALL),--all,--run-id $(RUN_ID)) --out $(RUNS_DIR)

report: ## re-render a local run: make report RUN=runs/<run-id>
	@test -n "$(RUN)" || { echo "usage: make report RUN=runs/<run-id>"; exit 2; }
	python3 analysis/coldstart_report.py $(RUN)/trace -o $(RUN)

compare: ## diff two local runs: make compare BASE=runs/a NEW=runs/b
	@test -n "$(BASE)" -a -n "$(NEW)" || { echo "usage: make compare BASE=runs/a NEW=runs/b"; exit 2; }
	python3 analysis/coldstart_report.py --compare $(BASE)/trace $(NEW)/trace

lint: ## syntax-check every script and module
	@for f in coldstart/*.py analysis/*.py tests/*.py; do python3 -c "import ast,sys;ast.parse(open('$$f').read())" || exit 1; done
	@for f in scripts/*.sh tests/*.sh; do bash -n $$f || exit 1; done
	@echo "lint ok"

clean: ## remove local self-test output and generated manifests
	rm -rf .selftest manifests/generated
