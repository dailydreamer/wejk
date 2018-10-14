## Usage

start test server

```sh
source ./set_env.sh
FLASK_APP=wsgi flask run -p 8000
```

test upload_csv

```sh
curl -X POST -H 'Content-Type: multipart/form-data; charset=utf-8' -F 'csv_file=@data/test_upload_csv.csv' http://localhost:8000/api/v1/upload_csv
```

test upload_json

```sh
curl -X POST -H 'Content-Type: application/json; charset=utf-8' -d @data/test_upload_json.json http://localhost:8000/api/v1/upload_json
```

test predict_month_sku

```sh
curl -X POST -H 'Content-Type: application/json; charset=utf-8' -d @data/test_predict_month_sku.json http://localhost:8000/api/v1/predict_month_sku
```

export conda environment

```sh
conda env export > environment.yml
```

import conda environment

```sh
conda env create -f environment.yml -n iFashion
```

run with gunicorn

```sh
gunicorn wsgi:app -c gun_conf.py
```

build docker image

```sh
docker build -t dailydreamer/ifashion .
```

test docker image

```sh
docker run -it --rm \
  --name ifashion \
  -p 8000:8000 \
  -e SECRET_KEY \
  -e MYSQL_URI \
  dailydreamer/ifashion
```

tag release to trigger docker image build
```sh
git tag release-va.b.c
```