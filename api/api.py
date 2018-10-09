import traceback
from flask import Blueprint, request, jsonify, current_app

from . import utils
from . import db
from . import model

API_VERSION = 'v1'

bp = Blueprint('api', __name__, url_prefix='/api/{}'.format(API_VERSION))

@bp.route('/upload_csv', methods=['POST'])
def upload_csv():
    current_app.logger.info('{} request received from: {}'.format(
        request.method, request.remote_addr))
    if 'csv_file' not in request.files:
        current_app.logger.error('No csv_file field')
        return jsonify(error='No csv_file field'), 400
    csv_file = request.files['csv_file']
    if csv_file.filename == '':
        current_app.logger.error('No csv file selected')        
        return jsonify(error='No csv file selected'), 400
    if not (csv_file and '.' in csv_file.filename and csv_file.filename.rsplit('.', 1)[1].lower() == 'csv'):
        current_app.logger.error('Provided file is not csv file')        
        return jsonify(error='Provided file is not csv file'), 400
    try:
        df = utils.read_csv(csv_file)
        for index, row in df.iterrows():
            row_dict = row.to_dict()
            db.create_record(row_dict)
    except ValueError as e:
        current_app.logger.error('Upload csv error: {}\n{}'.format(e, traceback.format_exc()))
        return jsonify(error='Read csv file error: {}'.format(e)), 422
    current_app.logger.info('Upload file {} success'.format(csv_file.filename))
    return jsonify(error=None)

@bp.route('/predict_month_sku', methods=['POST'])
def predict_month_sku():
    para = request.get_json()
    required_fileds = {'tenant_id', 'para_list'}
    if not para:
        return jsonify(error='Problems parsing JSON'), 400
    if not required_fileds.issubset(set(para.keys())):
        return jsonify(error='Not enough parameters'), 422
    import pandas as pd
    input_df = pd.DataFrame(para['para_list'])
    input_df['tenant_id'] = para['tenant_id']
    df = model.predict_month_sku(input_df)
    # TODO return json
    return 'fake'