import traceback
import pandas as pd
import numpy as np
from api.index import create_app
from api.db import create_record
from api.utils import read_csv

def upload_csv(csv_file):
    try:
        df = read_csv(csv_file)
        for index, row in df.iterrows():
            row_dict = row.to_dict()
            create_record(row_dict)
    except ValueError as e:
        print(e)
        traceback.print_exc()
    print('upload success')

if __name__ == '__main__':
    app = create_app()
    with app.app_context():
        upload_csv('./data/test_upload_csv.csv')
