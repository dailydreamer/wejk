## Usage

start test server

```sh
source ./set_env.sh
FLASK_APP=api.index flask run -p 5000
```

test upload_csv

```sh
curl -X POST -H 'Content-Type: multipart/form-data' -F 'csv_file=@data/test_upload_csv.csv' http://localhost:5000/api/v1/upload_csv
```

test predict_month_sku

```sh
curl -X POST -H "Content-Type: application/json" -d @data/test_predict_month_sku.json http://localhost:5000/api/v1/predict_month_sku
```

export conda environment

```sh
conda env export > environment.yml
```

import conda environment

```sh
conda env create -f environment.yml -n iFashion
```