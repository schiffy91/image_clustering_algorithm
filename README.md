# README

## Setup

### Docker
1. Install & run Docker Desktop
2. Install VSCode & open `image-clustering-algorithm.code-workspace`
3. Install the Docker and Dev Container Extensions
4. `ctrl-shift-p` (or `cmd-shift-p`) and select `Dev Containers: Rebuild and Reopen in Container` 
5. Open `app.ipynb`, select the kernel by choosing `Python Environments` and the global python interpreter (`/bin/python`). Run the notebook as you would.

### Docker-less
1. `export ICA_PATH=~/.local/share/virtualenvs/ica`
2. `cd src && python3 -m venv $ICA_PATH && source $ICA_PATH/bin/activate`
3. `pip install .`

## Inputs and Outputs
The algorithm is hard-coded to look for a flat directory of images at `~/Downloads/JPGs`. Despite the name, the algorithm can also work on PNGs, TIFFs, and more. 

The algorithm can output three items:
1. A seralized list of features for the directory.
2. A serialized list of clusters for the directory.
3. The images placed in their own clusters. To write the sorted images to disk, uncomment `#ica.save_clustered_images("~/Downloads/ClusteredJPGs")` in `app.ipynb`; otherwise, you can use the notebook's cells to view 5 photos in each cluster to determine the algorithm's performance.

### Optimal Number of Clusters
The algorithm defaults to 50 clusters. You can change `find_optimal_num_clusters` to `True` in `app.ipynb` (`ica.compute_clusters(max_clusters, find_optimal_num_clusters=False)`) if you want it enumerate K-means for cluster size [2...n] (and choose the optimal cluster size). Note: This will take a long time.


## GPU Acceleration
The code work on Mac via Metal and Windows / Linux via CUDA. If you're running out of memory, change `batch_size = 16`, `num_workers = 32`.