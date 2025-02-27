FROM ncbi/blast:latest

COPY cdhit cdhit

# Install dependencies
RUN apt-get update
RUN apt-get -y install curl
RUN apt install -y zlib1g-dev
RUN apt-get install -y python3
WORKDIR cdhit
RUN make
RUN apt install cd-hit
WORKDIR /blast

COPY run_cdhit.py run_cdhit.py

CMD [ "python3", "run_cdhit.py" ]



