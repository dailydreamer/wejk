import traceback
from flask import Blueprint, request, jsonify, current_app

from .utils import read_csv
from .db import create_record, required_record_mapping
from .model import predict_monthly_sku, predict_daily_sku

API_VERSION = 'v1'

bp = Blueprint('api', __name__, url_prefix='/api/{}'.format(API_VERSION))

@bp.route('/upload_csv', methods=['POST'])
def api_upload_csv():
    current_app.logger.info('upload_csv: {} request received from: {}'.format(
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
        df = read_csv(csv_file)
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            create_record(row_dict)
    except ValueError as e:
        current_app.logger.error('Upload csv error: {}\n{}'.format(e, traceback.format_exc()))
        return jsonify(error='Read csv file error: {}'.format(e)), 422
    current_app.logger.info('Upload file {} success'.format(csv_file.filename))
    return jsonify(error=None)

@bp.route('/upload_json', methods=['POST'])
def api_upload_json():
    current_app.logger.info('upload_json: {} request received from: {}'.format(
        request.method, request.remote_addr))
    para = request.get_json(silent=True)
    required_fileds = required_record_mapping.keys()
    if not para:
        return jsonify(error='Problems parsing JSON'), 400
    if not set(required_fileds).issubset(set(para.keys())):
        return jsonify(error='Not enough parameters'), 422    
    try:
        create_record(para)
    except ValueError as e:
        current_app.logger.error('Upload json error: {}\n{}'.format(e, traceback.format_exc()))
        return jsonify(error='Upload json error: {}'.format(e)), 422
    current_app.logger.info('Upload json {} success')
    return jsonify(error=None)

@bp.route('/predict_month_sku', methods=['POST'])
def api_predict_month_sku():
    para = request.get_json(silent=True)
    required_fileds = {'tenant_id', 'para_list'}
    if not para:
        return jsonify(error='Problems parsing JSON'), 400
    if not set(required_fileds).issubset(set(para.keys())):
        return jsonify(error='Not enough parameters'), 422
    import pandas as pd
    input_df = pd.DataFrame(para['para_list'])
    input_df['tenant_id'] = para['tenant_id']
    df = predict_monthly_sku(para['tenant_id'], input_df)

    df = df.drop(['tenant_id'], axis = 1)
    res = {
        'tenant_id': para['tenant_id'],
        'para_list': df.to_dict('records')
    }
    return jsonify(res)

@bp.route('/predict_day_sku', methods=['POST'])
def api_predict_day_sku():
    para = request.get_json(silent=True)
    required_fileds = {'tenant_id', 'para_list'}
    if not para:
        return jsonify(error='Problems parsing JSON'), 400
    if not set(required_fileds).issubset(set(para.keys())):
        return jsonify(error='Not enough parameters'), 422
    import pandas as pd
    input_df = pd.DataFrame(para['para_list'])
    input_df['tenant_id'] = para['tenant_id']
    df = predict_daily_sku(para['tenant_id'], input_df)
    
    df = df.drop(['tenant_id'], axis = 1)
    res = {
        'tenant_id': para['tenant_id'],
        'para_list': df.to_dict('records')
    }
    return jsonify(res)
