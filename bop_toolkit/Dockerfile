# Use a base image that includes Conda and other required tools
FROM continuumio/miniconda3:latest

# Copy the environment.yaml file into the container
COPY environment.yaml /tmp/environment.yaml



# Create the Conda environment and activate it
RUN conda env create -f /tmp/environment.yaml

# Activate the Conda environment
ENV PATH /opt/conda/envs/myenv/bin:$PATH

#COPY bop_toolkit /workspace/bop_toolkit
