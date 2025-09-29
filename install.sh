conda create -n gs-scale python=3.10 -y
conda activate gs-scale
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install intel-extension-for-pytorch==2.2.0
cd examples && pip install -r requirements.txt && cd ..
pip install .
cd cpu_adam && python setup.py install
