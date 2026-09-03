#!/usr/bin/env bash
# Produce the tensorizer artifact that --load-format tensorizer needs.
#
# Unlike every other load format in reports/loader-comparison-32b.txt, tensorizer
# cannot read the HF safetensors checkpoint. It needs a .tensors file written by
# vLLM itself: the model is instantiated, its post-processed state dict is
# serialized, and the result is specific to this vLLM build, this dtype and this
# tensor-parallel size. So the artifact is a second full copy of the weights
# (61 GiB for Qwen3-32B) and producing it costs a full model load.
#
# That is the honest framing for any boot number measured against it: the fast
# load is real, and it is paid for once, offline, per (model, vLLM build, TP).
#
#   CS_NS=my-ns scripts/tensorize-artifact.sh
set -uo pipefail
NS="${CS_NS:-}"; POD="${CS_POD:-vllm-coldstart}"
MODEL="${CS_MODEL:-Qwen/Qwen3-32B}"; MML="${CS_MML:-8192}"
DIR="${CS_TZROOT:-/cache/tz}"; SUFFIX="${CS_TZSUFFIX:-v1}"
LP="${CS_LP:-/cache/lp}"

kubectl ${NS:+-n "$NS"} exec "$POD" -c vllm -- bash -lc "
set -o pipefail
mkdir -p $DIR
cat > /tmp/tzserialize.sh <<'EOS2'
#!/bin/bash
out=/tmp/tz-serialize.txt; : > \$out
# bash SECONDS rather than /usr/bin/time: the vLLM image has no GNU time.
t0=\$SECONDS
{ python3 /vllm-workspace/examples/features/tensorize_vllm_model.py \\
    --model $MODEL --max-model-len $MML \\
    serialize --serialized-directory $DIR --suffix $SUFFIX ; echo \"EXIT=\$?\"; } >> \$out 2>&1
echo \"WALL=\$((SECONDS-t0))s\" >> \$out
echo SERIALIZEDONE >> \$out
EOS2
chmod +x /tmp/tzserialize.sh
env PYTHONPATH=$LP CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/cache/pycache \
  nohup /tmp/tzserialize.sh >/tmp/tzserialize.log 2>&1 &
echo started
"
echo "watch:  kubectl ${NS:+-n $NS} exec $POD -c vllm -- tail -f /tmp/tz-serialize.txt"
