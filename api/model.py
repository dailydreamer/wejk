from datetime import datetime
from sklearn.externals import joblib

def train():
    pass

def load_model(tenant_id, date):
    """
    Args:
        tenant_id(string)
        date(datetime)
    """
    model = joblib.load('{}-{}.joblib'.format(tenant_id, date.date()))
    # TODO load from oss
    return model 

def save_model(model, tenant_id):
    today = datetime.now().date()
    joblib.dump(model, '{}-{}.joblib'.format(tenant_id, today))
    # TODO save to oss

def predict_month_sku(df):
    """
    predict
    Args:
        df(dataframe)
    Returns:
        df(dataframe): dataframe of predict result
    """
    pass
