FROM ubuntu:16.04

RUN apt-get update && apt-get install -y wget bzip2
RUN wget --quiet http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O ~/miniconda.sh && \ 
  bash ~/miniconda.sh -b -p /opt/conda && \ 
  rm ~/miniconda.sh
ENV PATH /opt/conda/bin:${PATH}

COPY . /app
WORKDIR /app

RUN conda update conda -y && \
    conda env create -f environment.yml -n wejk
ENV PATH /opt/conda/envs/wejk/bin:$PATH

EXPOSE 8000

CMD ["gunicorn", "wsgi:app", "-c", "gun_conf.py"]