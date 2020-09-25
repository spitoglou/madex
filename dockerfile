FROM continuumio/miniconda3:latest

# Install extra packages if required
# RUN apt-get update && apt-get install -y \
#     xxxxxx \
#     && rm -rf /var/lib/apt/lists/*

# Add the user that will run the app (no need to run as root)
RUN groupadd -r csuser && useradd -r -g csuser csuser

WORKDIR /app

# Install myapp requirements
COPY environment.yml /app/environment.yml
# RUN conda config --add channels conda-forge \
#     && conda env create -n myapp -f environment.yml \
#     && rm -rf /opt/conda/pkgs/*
# RUN conda create -n new_metric 

RUN conda env create -f environment.yml \
    # && source activate new_metric \
    # numpy==1.16.3 \
    # pandas==0.24.2 \
    # tini==0.18.0 \
    && conda clean -afy
    # && find /opt/conda/ -follow -type f -name '*.a' -delete \
    # && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    # && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    # && find /opt/conda/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete

# activate the myapp environment
RUN echo "source activate new_metric" > ~/.bashrc
ENV PATH /opt/conda/envs/new_metric/bin:$PATH

# Install myapp
COPY . /app/
RUN chown -R csuser:csuser /app/*


# ENTRYPOINT [ "/bin/bash" ]