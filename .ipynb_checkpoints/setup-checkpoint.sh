conda create -n venv
conda activate venv
pip install transformer_lens
pip install pytest
pip install ipykernel
python3 -m ipykernel install --prefix "${DL_ANACONDA_HOME}" --name venv --display-name venv