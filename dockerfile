FROM continuumio/miniconda3:latest

# Add the user that will run the app (no need to run as root)
RUN groupadd -r csuser && useradd -r -g csuser csuser

WORKDIR /app

# Install myapp requirements
COPY environment.yml /app/environment.yml

RUN conda env create -f environment.yml \
    && conda clean -afy

# activate the myapp environment
RUN echo "source activate new_metric" > ~/.bashrc
ENV PATH /opt/conda/envs/new_metric/bin:$PATH

# Install myapp
COPY . /app/
RUN chown -R csuser:csuser /app/*

# ENTRYPOINT [ "/bin/bash" ]