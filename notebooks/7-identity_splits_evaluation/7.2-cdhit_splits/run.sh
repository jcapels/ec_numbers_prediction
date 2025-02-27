

podman build . -t enzymes_clusters
podman run -v $(pwd)/data/:/blast/data/:Z -d --name enzymes_clusters enzymes_clusters 