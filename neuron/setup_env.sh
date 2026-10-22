#!/usr/bin/env bash
# Python env for the inf2 port, layered on the Neuron DLAMI's torch-neuronx venv (torch 2.14 + NKI 0.7, SDK 2.33):
# a plain venv whose site-packages chains to the DLAMI venv (.pth site.addsitedir), plus this repo (editable, no deps:
# its pinned CUDA stack does not apply) and the few CPU-side packages the upstream loader/prompt code imports.
# Usage: ./neuron/setup_env.sh [venv dir, default ~/venvs/qtts] ; then: source <venv>/bin/activate
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${1:-$HOME/venvs/qtts}
BASE=${NEURON_VENV:-/opt/aws_neuronx_venv_torch_neuronx_2_14_0}
SP=lib/python3.12/site-packages

python3.12 -m venv "$VENV"
echo "import site; site.addsitedir('$BASE/$SP')" > "$VENV/$SP/zz_neuron_base.pth"
"$VENV/bin/pip" install -q --no-deps -e .
"$VENV/bin/pip" install -q qwen-tts==0.1.1 --no-deps
"$VENV/bin/pip" install -q transformers==4.57.3 tokenizers==0.22.2 accelerate==1.12.0 huggingface_hub==0.36.2 \
    librosa==1.0.0 soundfile==0.14.0 sox==1.5.0 onnxruntime==1.31.0 uvicorn==0.52.3 websockets==17.0.1 regex
# No torchaudio wheel exists for torch 2.14 (Neuron); only qwen_tts's unused 25 Hz tokenizer imports it.
mkdir -p "$VENV/$SP/torchaudio"
cat > "$VENV/$SP/torchaudio/__init__.py" <<'EOF'
"""Stub: no torchaudio wheel exists for torch 2.14 (Neuron). Only qwen_tts's unused
25Hz tokenizer imports it; the 12Hz model never calls it. Any real use fails loudly."""
__version__ = "0.0.0+neuron-stub"
EOF
"$VENV/bin/python" -c "import torch, nki, nari_qwen3_tts; print('ok: torch', torch.__version__, '| nki', nki.__version__)"
