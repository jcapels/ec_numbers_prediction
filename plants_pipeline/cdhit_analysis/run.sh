

podman build . -t cdhit_plants
podman run -v $(pwd)/data/:/blast/data/:Z -d --name cdhit_plants cdhit_plants