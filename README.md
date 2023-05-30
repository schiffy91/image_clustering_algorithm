# README

## Docker
1. Install & run Docker Desktop
2. Install VSCode & open `image-clustering-algorithm.code-workspace`
3. Install Docker and Dev Container Extensions
4. `ctrl-shift-p` (or `cmd-shift-p`) and select `Dev Containers: Rebuild and Reopen in Container` 
5. Open `app.ipynb`, select the global python interpreter, and run the notebook

## Docker-less
1. `export ICA_PATH=~/.local/share/virtualenvs/ica`
2. `cd src && python3 -m venv $ICA_PATH && source $ICA_PATH/bin/activate`
3. `pip install .`

## Images
Right now, the algorithm is coded to look for a flat directory of images at `~/Downloads/JPGs`

## GPU
Should work on Mac via Metal and Windows / Linux via CUDA. If you're running out of memory, change batch_size = 16, num_workers = 32.