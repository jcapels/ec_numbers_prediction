

podman build . -t cdhit_swiss_prot
podman run -v $(pwd)/data/:/blast/data/:Z -d --name cdhit_swiss_prot cdhit_swiss_prot 