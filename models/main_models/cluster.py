from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse
import numpy as np
import json
from collections import defaultdict
from sklearn.metrics import davies_bouldin_score
import h5py
from sklearn.metrics import silhouette_score


model = SentenceTransformer("all-mpnet-base-v2") # all-MiniLM-L6-v2, , all-distilroberta-v1, sentence-t5-base,all-MiniLM-L12-v2,all-mpnet-base-v2 

READ = False

if READ:
    path = "/users/ajaafar/data/shared/lanmp/sim_dataset.hdf5"
    commands=[]
    # Open the HDF5 file
    with h5py.File(path, 'r') as hdf_file:
        # Iterate through each trajectory group
        for trajectory_name, trajectory_group in hdf_file.items():
            # Iterate through each timestep group within the trajectory
            for timestep_name, timestep_group in trajectory_group.items():
                # Read and decode the JSON metadata
                metadata = json.loads(timestep_group.attrs['metadata'])
                commands.append(metadata['nl_command'])
                break

    np.save("sim_commands.npy", commands)
else:
    commands = np.load('sim_commands.npy', allow_pickle=True).tolist()

# Convert the commands to embeddings
embeddings = model.encode(commands)

# Compute the cosine similarity matrix
# sim_mat = cosine_similarity(embeddings)
# print(sim_mat)

# Apply Agglomerative Clustering
clustering = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.3)
clusters = clustering.fit_predict(embeddings)

# Calculate the silhouette score
silhouette_avg = silhouette_score(embeddings, clusters, metric='cosine')
print(f'Silhouette Score: {silhouette_avg}')

# Calculate the Davies-Bouldin index
db_index = davies_bouldin_score(embeddings, clusters)
print(f'Davies-Bouldin Index: {db_index}')

# Create a defaultdict with lists as default values
cluster_dict = defaultdict(list)

# Populate the dictionary
for string, cluster_id in zip(commands, clusters):
    cluster_dict[int(cluster_id)].append(string)

cluster_dict = dict(cluster_dict)
print(f'num clusters: {len(cluster_dict.keys())}')

#save dict
# with open('cluster_dict.json', 'w') as f:
#     json.dump(cluster_dict, f)

# Find the cluster with the longest list
sorted_clusters = sorted(cluster_dict, key=lambda k: len(cluster_dict[k]), reverse=True)

tot = 0
# for i in [0,1,2,3,4,5,6,7,8,9]: #low number of clusters #240
# for i in range(14, 106): #high number of clusters #238
for i in range(10,14): #test 46
    print(f"{len(cluster_dict[sorted_clusters[i]])} elements.")
    tot += len(cluster_dict[sorted_clusters[i]])

print(f"{tot} total")
breakpoint()

# Apply t-SNE for dimensionality reduction
reduced_embeddings = TSNE(n_components=2, metric='cosine', perplexity=175).fit_transform(embeddings)

# Plot the t-SNE results
plt.figure(figsize=(10, 8))
for cluster in np.unique(clusters):
    cluster_points = reduced_embeddings[clusters == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')

    # Calculate the ellipse for each cluster
    if len(cluster_points) > 1:  # Ellipse requires at least 2 points
        cov = np.cov(cluster_points, rowvar=False)
        mean = np.mean(cluster_points, axis=0)

        # Eigen decomposition of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

        # Compute width, height and angle of ellipse
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues)

        # Draw the ellipse
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='black', facecolor='none', lw=2)
        plt.gca().add_patch(ellipse)

plt.title('t-SNE with Cluster Ellipses')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.grid(True)
plt.show()