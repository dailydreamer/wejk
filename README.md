## Deploy

### Set up database

connect to local databse root user
```sh
mysql -u root -p
```

create test uesr in mysql cmd
```sql
CREATE USER 'test'@'localhost' IDENTIFIED BY 'password';
```

grant privileges to test user in mysql cmd
```sql
GRANT ALL PRIVILEGES ON *.* TO 'test'@'localhost' WITH GRANT OPTION;
```

create test databse in mysql cmd
```sql
CREATE DATABASE test
  DEFAULT CHARACTER SET utf8
  DEFAULT COLLATE utf8_general_ci;
```

Set up databse connection parameters from server
```sh
source ./set_env.sh
```

create tables
```sh
python init_db.py
```

### deploy with conda for test only

start test server with conda environment

```sh
FLASK_APP=wsgi flask run -p 8000
```

export conda environment

```sh
conda env export > environment.yml
```

import conda environment

```sh
conda env create -f environment.yml -n wejk
```

run with gunicorn

```sh
gunicorn wsgi:app -c gun_conf.py
```

### deploy with docker 

build docker image

```sh
docker build -t dailydreamer/wejk .
```

run docker image

```sh
docker run -it --rm \
  --name ifashion \
  -p 8000:8000 \
  -e SECRET_KEY \
  -e MYSQL_URI \
  dailydreamer/wejk
```

tag release to trigger docker image build on Aliyun server
```sh
git tag release-va.b.c
```

## Test

upload csv

```sh
python upload_csv.py -p 'path to csv'
```

upload_json

```sh
curl -X POST -H 'Content-Type: application/json; charset=utf-8' -d @data/test_upload_json.json http://localhost:8000/api/v1/upload_json
```

train

```sh
python train.py -m 'm'/'d' -t 'tenant_id'
```

predict_month_sku

```sh
curl -X POST -H 'Content-Type: application/json; charset=utf-8' -d @data/test_predict_month_sku.json http://localhost:8000/api/v1/predict_month_sku
```