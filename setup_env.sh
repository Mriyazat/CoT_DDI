set -e

module load StdEnv/2023
module load gcc/12.3 python/3.11 cuda/12.2 arrow opencv/4.12.0

rm -rf $HOME/ddi_venv
virtualenv --no-download $HOME/ddi_venv
source $HOME/ddi_venv/bin/activate
pip install --no-index --upgrade pip

# Bypass CC dummy-wheel traps for opencv and pyarrow
pip install --no-index --no-deps --ignore-installed opencv-python-headless

mkdir -p /tmp/_cv_stub
echo "from setuptools import setup; setup(name='opencv-noinstall', version='9999+dummy.computecanada')" > /tmp/_cv_stub/setup.py
pip install --force-reinstall /tmp/_cv_stub
rm -rf /tmp/_cv_stub

mkdir -p /tmp/_pa_stub
echo "from setuptools import setup; setup(name='pyarrow-noinstall', version='9999+dummy.computecanada')" > /tmp/_pa_stub/setup.py
pip install --force-reinstall /tmp/_pa_stub
rm -rf /tmp/_pa_stub

pip install --no-index --no-deps --ignore-installed pyarrow

# Expose real pyarrow from the arrow module
ARROW_SP=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/arrow/23.0.1/lib/python3.11/site-packages
echo "$ARROW_SP" > "$(python -c 'import site; print(site.getsitepackages()[0])')"/cc_arrow.pth

# Install packages
pip install --no-index vllm
pip install --no-index accelerate datasets peft bitsandbytes
pip install --no-index --no-deps torch==2.10.0
pip install --no-index scikit-learn matplotlib seaborn jsonlines wandb
pip install trl
pip install rouge-score 2>/dev/null || echo "rouge-score not available via CC — will install from PyPI"
pip install bert-score 2>/dev/null || echo "bert-score not available via CC — will install from PyPI"
pip install rdkit-pypi 2>/dev/null || echo "rdkit not available — ML baseline will be skipped"

# HF cache on scratch
export HF_HOME=$SCRATCH/.cache/huggingface
mkdir -p $HF_HOME

echo ""
echo "============================================================"
echo " Setup complete! Verify with:"
echo "   source activate_env.sh"
echo "   python -c \"import vllm, torch, transformers, peft, datasets, trl; print('All good')\""
echo ""
echo " Then download models:"
echo "   bash scripts/download_models.sh"
echo "============================================================"
