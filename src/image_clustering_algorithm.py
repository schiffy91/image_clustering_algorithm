import os
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import InceptionResnetV1
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import glob
import pickle
import warnings

batch_size = 16
num_workers = 32
warnings.filterwarnings("ignore")

class ImageDataset(Dataset):
    def __init__(self, image_directory):
        super(ImageDataset, self,).__init__()
        self.image_directory = os.path.expanduser(image_directory)
        file_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif")
        self.image_paths = []
        for extension in file_extensions:
            self.image_paths.extend(glob.glob(os.path.join(self.image_directory, extension)))
        self.transform = transforms.Compose([transforms.Resize((224, 224)), 
                         transforms.ToTensor(), 
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)

class ImageClusteringAlgorithm:

    def __init__(self):
        self.version = "0.1"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.facenet_model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.resnet_model = models.resnet50(pretrained=True).eval().to(self.device)        
    
    def compute_features(self, image_directory):
        self.image_dataset = ImageDataset(image_directory)
        image_loader = DataLoader(self.image_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True)
        features = []
        with torch.no_grad():
            for image in tqdm(image_loader, desc="Computing features..."):
                tensor = image.to(self.device)
                resnet_features = self.resnet_model(tensor)
                face_features = self.facenet_model(tensor)
                features.append(np.concatenate((face_features.cpu().detach().numpy(), resnet_features.cpu().detach().numpy()), axis=1))
                torch.cuda.empty_cache()
        self.features = np.vstack(features)

    def compute_clusters(self, max_clusters, find_optimal_num_clusters=True):
        max_score = 0
        min_clusters = max_clusters
        if find_optimal_num_clusters:
            min_clusters = 2
        for i in tqdm(range(min_clusters, max_clusters + 1), "Calculating optimal clusters..."):
            kmeans = KMeans(n_clusters=i, random_state=42)
            labels = kmeans.fit_predict(self.features)
            silhouette_avg = silhouette_score(self.features, labels)
            print(f"Silhouette score: {silhouette_avg:.4f}")
            if silhouette_avg >= max_score:
                print(f"new best cluster: {i}, silhouette score: {silhouette_avg:.4f}")
                max_score = silhouette_avg
                self.num_clusters = i
                self.labels = labels
                self.clusters = [[] for _ in range(self.labels.max() + 1)]
                for idx, label in enumerate(self.labels):
                    self.clusters[label].append(self.image_dataset.image_paths[idx])

    def save_clustered_images(self, output_directory, dry_run=False):
        output_directory = os.path.expanduser(output_directory)
        os.makedirs(output_directory, exist_ok=True)
        for cluster_id in tqdm(range(self.num_clusters), "Saving clusters..."):
            i = 0
            cluster_name_zfill = len(str(self.num_clusters))
            cluster_name = format(f"{cluster_id}".zfill(cluster_name_zfill))
            cluster_element_zfill = len(str(f"{len(self.clusters[cluster_id])}"))
            for image_path in self.clusters[cluster_id]:
                cluster_element = format(f"{i}".zfill(cluster_element_zfill))
                new_file_name = os.path.join(output_directory, f"cluster_{cluster_name}_{cluster_element}{os.path.splitext(image_path)[1]}")
                i = i + 1
                if dry_run:
                    print(new_file_name)
                else:
                    shutil.copyfile(image_path, new_file_name)

    def save_features(self, file_name):
        file_name = os.path.expanduser(file_name)
        data = { "features": self.features, "image_dataset": self.image_dataset, "version": self.version }
        with open(file_name, "wb") as file: 
            pickle.dump(data, file)
    
    def load_features(self, file_name):
        file_name = os.path.expanduser(file_name)
        with open(file_name, "rb") as file:
            data = pickle.load(file)
        self.features = data["features"]
        self.image_dataset = data["image_dataset"]

    def save_clusters(self, file_name):
        file_name = os.path.expanduser(file_name)
        data = { "num_clusters": self.num_clusters, "clusters": self.clusters, "labels": self.labels, "version": self.version }
        with open(file_name, "wb") as file: 
            pickle.dump(data, file)

    def load_clusters(self, file_name):
        with open(file_name, "rb") as file:
            data = pickle.load(file)
        self.num_clusters = data["num_clusters"]
        self.clusters = data["clusters"]
        self.labels = data["labels"]