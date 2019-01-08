import os
from datetime import datetime
from sklearn.externals import joblib
from .db import get_records_periods, get_records
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import numpy as np

def load_model(tenant_id, m_or_d):
    """
    Args:
        tenant_id(string)
        m_or_d(string): m stands for monthly, d for daily
    Returns:
        model: sklearn model
        encoder: sklearn encoder
        start_year: start year of data for indexing
    """
    
    model = joblib.load('model/'+'{}-{}.model.joblib'.format(tenant_id, m_or_d))
    encoder_store = joblib.load('model/'+'{}-{}.encoder_store.joblib'.format(tenant_id, m_or_d))
    encoder_cate = joblib.load('model/'+'{}-{}.encoder_cate.joblib'.format(tenant_id, m_or_d))
    encoder_size = joblib.load('model/'+'{}-{}.encoder_size.joblib'.format(tenant_id, m_or_d))
    encoder_color = joblib.load('model/'+'{}-{}.encoder_color.joblib'.format(tenant_id, m_or_d))
    start_year = joblib.load('model/'+'{}-{}.start_year.joblib'.format(tenant_id, m_or_d))
    return model, encoder_store, encoder_cate, encoder_size, encoder_color, start_year
    
def save_model(tenant_id, m_or_d, model, encoder_store, encoder_cate, encoder_color, encoder_size, start_year):
    today = datetime.now().date()
    if not os.path.exists('model'):
        os.makedirs('model')
    # save and override model
    joblib.dump(model, 'model/'+'{}-{}-{}.model.joblib'.format(tenant_id, today, m_or_d))
    joblib.dump(model, 'model/'+'{}-{}.model.joblib'.format(tenant_id, m_or_d))
    # save and override encoder
    joblib.dump(encoder_store, 'model/'+'{}-{}-{}.encoder_store.joblib'.format(tenant_id, today, m_or_d))
    joblib.dump(encoder_store, 'model/'+'{}-{}.encoder_store.joblib'.format(tenant_id, m_or_d))
    
    joblib.dump(encoder_cate, 'model/'+'{}-{}-{}.encoder_cate.joblib'.format(tenant_id, today, m_or_d))
    joblib.dump(encoder_cate, 'model/'+'{}-{}.encoder_cate.joblib'.format(tenant_id, m_or_d))
    
    joblib.dump(encoder_color, 'model/'+'{}-{}-{}.encoder_color.joblib'.format(tenant_id, today, m_or_d))
    joblib.dump(encoder_color, 'model/'+'{}-{}.encoder_color.joblib'.format(tenant_id, m_or_d))
    
    joblib.dump(encoder_size, 'model/'+'{}-{}-{}.encoder_size.joblib'.format(tenant_id, today, m_or_d))
    joblib.dump(encoder_size, 'model/'+'{}-{}.encoder_size.joblib'.format(tenant_id, m_or_d))
    # save and override start_year
    joblib.dump(start_year, 'model/'+'{}-{}-{}.start_year.joblib'.format(tenant_id, today, m_or_d))
    joblib.dump(start_year, 'model/'+'{}-{}.start_year.joblib'.format(tenant_id, m_or_d))

## Annie TO DO:
# /1. Delete drop columns
# /2. Optimize encoder
# /3. Check predict_daily_sku
# /4. Add date (end)
# 5. Save predicted results function
# 6. Test function, 'RMSE-QUANTITY:', np.sqrt(metrics.mean_squared_error(y_test['quantity'],y_predict[:,0]))
# Stage2:
# 5. Optimize models(cross-validation, ensemble learning; try other xgboost)
# 6. Predict price
def train_daily(tenant_id):
    # 1-Load data from db
    # end = datetime.now()
    # df = get_records_periods(tenant_id, start, end)
    df = get_records(tenant_id)
    
    # 2-Data preprocessing (#Data pipeline): 
    # df_dropped = dropColumns(df)
    df = dropColumns(df)
    
    # df_encoded = encodeData(df_dropped)
    str_cols = ['store_id','category', 'size', 'color']
    df[str_cols] = df[str_cols].astype('str')
    
    le_store = preprocessing.LabelEncoder().fit(df['store_id'])
    df['store_id_index'] = le_store.transform(df['store_id'])
    le_cate = preprocessing.LabelEncoder().fit(df['category'])
    df['category_index'] = le_cate.transform(df['category'])
    le_size = preprocessing.LabelEncoder().fit(df['size'])
    df['size_index'] = le_size.transform(df['size'])
    le_color = preprocessing.LabelEncoder().fit(df['color'])
    df['color_index'] = le_color.transform(df['color'])
    
    # df_parsed = parseTime(df_encoded)
    df['year'] = df['sale_date'].dt.year
    df['month'] = df['sale_date'].dt.month
    df['weekday'] = df['sale_date'].dt.weekday
    start_year = min(df.year)  #SAVE
    df['month_index'] = (df['year']-start_year)*12 + df['month'] - 1
    df['day'] = df['sale_date'].dt.day
    
    #df_featured = featureEng(df_parsed)
    df_featured = featureEng(df)
    
    #3.1 train daily sales
    #drop columns for training
    train_col = ['store_id_index', 'category_index', 'size_index','color_index', 
                 'year', 'month', 'day', 'weekday', 'month_index', 
                 'daily_sku_sales', 'daily_sku_revenue']
    xcol = [c for c in train_col if c not in ['daily_sku_sales', 'daily_sku_revenue']]
    df_train = df_featured.copy().fillna(0)
    X_train = df_train[xcol]
    y_train = df_train[['daily_sku_sales', 'daily_sku_revenue']]
    
    regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=12, random_state=15))
    regr_multirf.fit(X_train, y_train)
    
    model_daily = regr_multirf   
    # save_model_daily(model_daily, tenant_id)
    save_model(tenant_id, 'd', model_daily, le_store, le_cate, le_color, le_size, start_year)
    
def train_monthly(tenant_id):
    # 1-Load data from db
    # end = datetime.now()
    # df = get_records_periods(tenant_id, start, end)
    df = get_records(tenant_id)
    
    # 2-Data preprocessing (#Data pipeline): 
    # df_dropped = dropColumns(df)
    df = dropColumns(df)
    
    # df_encoded = encodeData(df_dropped)
    str_cols = ['store_id','category', 'size', 'color']
    df[str_cols] = df[str_cols].astype('str')
    
    le_store = preprocessing.LabelEncoder().fit(df['store_id'])
    df['store_id_index'] = le_store.transform(df['store_id'])
    le_cate = preprocessing.LabelEncoder().fit(df['category'])
    df['category_index'] = le_cate.transform(df['category'])
    le_size = preprocessing.LabelEncoder().fit(df['size'])
    df['size_index'] = le_size.transform(df['size'])
    le_color = preprocessing.LabelEncoder().fit(df['color'])
    df['color_index'] = le_color.transform(df['color'])
    
    # df_parsed = parseTime(df_encoded)
    df['year'] = df['sale_date'].dt.year
    df['month'] = df['sale_date'].dt.month
    df['weekday'] = df['sale_date'].dt.weekday
    start_year = min(df.year)  #SAVE
    df['month_index'] = (df['year']-start_year)*12 + df['month'] - 1
    df['day'] = df['sale_date'].dt.day
    
    #df_featured = featureEng(df_parsed)
    df_featured = featureEng(df)
    
    #3.2 train monthly sales
    train_monthly_col = ['store_id_index', 'category_index', 'size_index','color_index',
                     'year', 'month', 'month_index', 
                     'monthly_sku_sales', 'monthly_sku_revenue']
    xcol = [c for c in train_monthly_col if c not in ['monthly_sku_sales', 'monthly_sku_revenue']]
    df_train = df_featured.copy().fillna(0)
    df_train = df_train[train_monthly_col].drop_duplicates()
    X_train = df_train[xcol]
    y_train = df_train[['monthly_sku_sales', 'monthly_sku_revenue']]
    
    regr_multirf_M = MultiOutputRegressor(RandomForestRegressor(max_depth=12, random_state=15))
    regr_multirf_M.fit(X_train, y_train)
    
    model_monthly = regr_multirf_M   
    # save_model_monthly(model_monthly, tenant_id)
    save_model(tenant_id, 'm', model_monthly, le_store, le_cate, le_color, le_size, start_year)

# Data pipeline
def dropColumns(df, thres = 2):
#     ori_columns = df.columns#to be deleted
#     drop_columns = []#to be deleted
#     df = df.drop(df[drop_columns], axis = 1)#to be deleted
    
    #WHEN column[0](tenant_id) is null
    drop_columns = []
    df = df.drop(['tenant_id'], axis = 1)
    #DROP UNIQUE VALUES<2 COLUMNS
    for columns in df:
        if df[columns].nunique() < thres:
            drop_columns.append(columns)
    df = df.drop(df[drop_columns], axis = 1)
    return df

def encodeData(df):##TO be optimized------------------------------------------'XC TODO: optimize 4 label encoder'
    str_cols = ['store_id','category', 'size', 'color']
    df[str_cols] = df[str_cols].astype('str')
    
#     # Encoding the variable
#     fit = df.apply(lambda x: d[x.name].fit_transform(x))
#     # Inverse the encoded
#     fit.apply(lambda x: d[x.name].inverse_transform(x))
#     # Using the dictionary to label future data
#     df.apply(lambda x: d[x.name].transform(x))
    
    le_store = preprocessing.LabelEncoder().fit(df['store_id'])
    df['store_id_index'] = le_store.transform(df['store_id'])
    le_cate = preprocessing.LabelEncoder().fit(df['category'])
    df['category_index'] = le_cate.transform(df['category'])
    le_size = preprocessing.LabelEncoder().fit(df['size'])
    df['size_index'] = le_size.transform(df['size'])
    le_color = preprocessing.LabelEncoder().fit(df['color'])
    df['color_index'] = le_color.transform(df['color'])
    return df
    
def parseTime(df_dropped):
    #parse time
    #CREATE TIME FEATURE: year, month, day, weekday, month_index, selling days, onshelf days
    df_dropped['year'] = df_dropped['sale_date'].dt.year
    df_dropped['month'] = df_dropped['sale_date'].dt.month
    df_dropped['weekday'] = df_dropped['sale_date'].dt.weekday
    #global startYear = min(df_dropped.year)  #SAVE AS DF
    df_dropped['start_year'] = min(df_dropped.year)
    df_dropped['month_index'] = (df_dropped['year']-df_dropped['start_year'])*12 + df_dropped['month'] - 1
#     df_dropped['onshelf_days'] = df_dropped['sale_date'] - df_dropped['hit_date']
#     df_dropped['onshelf_days'] = df_dropped['onshelf_days'].dt.days
    df_dropped['day'] = df_dropped['sale_date'].dt.day
    return df_dropped

def featureEng(df_dropped):
    #2 df processing: add feature, save data
    #add feature
        #AGGREGATE BY MONTH
    monthly_sku_sales = df_dropped.groupby(['store_id', 'category', 'size', 'color', 'year', 'month'], as_index=False)['quantity'].sum()
    monthly_sku_saledays = df_dropped.groupby(['store_id', 'category', 'size', 'color', 'year', 'month'], as_index=False)['sale_date'].agg({'sale_date': lambda x: (max(x) - min(x)).days})
    df_dropped['total_sale'] = df_dropped['quantity'] * df_dropped['sale_price']
    monthly_sku_revenue = df_dropped.groupby(['store_id', 'category', 'size', 'color', 'year', 'month'], as_index=False)['total_sale'].sum()
    monthly_sku_sales = monthly_sku_sales.rename(columns={'quantity':'monthly_sku_sales'})
    df_dropped = pd.merge(df_dropped, monthly_sku_sales, how = 'left', on=['store_id', 'category', 'size', 'color', 'year', 'month'])
    monthly_sku_saledays = monthly_sku_saledays.rename(columns={'sale_date':'monthly_sku_saledays'})
    df_dropped = pd.merge(df_dropped, monthly_sku_saledays, how = 'left', on=['store_id', 'category', 'size', 'color', 'year', 'month'])
    monthly_sku_revenue = monthly_sku_revenue.rename(columns={'total_sale': 'monthly_sku_revenue'})
    df_dropped = pd.merge(df_dropped, monthly_sku_revenue, how = 'left', on=['store_id', 'category', 'size', 'color', 'year', 'month'])
    
    ##AGGREGATE BY DAY
    daily_sku_sales = df_dropped.groupby(['store_id', 'category', 'size', 'color', 'year', 'month', 'day'], as_index=False)['quantity'].sum()
    daily_sku_saledays = df_dropped.groupby(['store_id', 'category', 'size', 'color', 'year', 'month', 'day'], as_index=False)['sale_date'].agg({'sale_date': lambda x: (max(x) - min(x)).days})
    #df_dropped['total_sale'] = df_dropped['quantity'] * df_dropped['sale_price']
    daily_sku_revenue = df_dropped.groupby(['store_id', 'category', 'size', 'color', 'year', 'month', 'day'], as_index=False)['total_sale'].sum()
    daily_sku_sales = daily_sku_sales.rename(columns={'quantity':'daily_sku_sales'})
    df_dropped = pd.merge(df_dropped, daily_sku_sales, how = 'left', on=['store_id', 'category', 'size', 'color', 'year', 'month', 'day'])
    daily_sku_saledays = daily_sku_saledays.rename(columns={'sale_date':'daily_sku_saledays'})
    df_dropped = pd.merge(df_dropped, daily_sku_saledays, how = 'left', on=['store_id', 'category', 'size', 'color', 'year', 'month', 'day'])
    daily_sku_revenue = daily_sku_revenue.rename(columns={'total_sale': 'daily_sku_revenue'})
    df_dropped = pd.merge(df_dropped, daily_sku_revenue, how = 'left', on=['store_id', 'category', 'size', 'color', 'year', 'month', 'day'])
    
    
    return df_dropped
 
def predict_daily_sku(tenant_id, df):
    """
    predict
    Args:
        df(dataframe)
    Returns:
        df(dataframe): dataframe of predict result
    """
    
    #Encode
    df_ori_columns = df.columns
    #Rename
    df=df.rename(index=str, columns={"predict_year": "year", "predict_month": "month", "predict_day": "day", "predict_weekday": "weekday"})
    
    today = datetime.now().date()
    #pred_model = load_model_daily(tenant_id, today)
    pred_model, le_store, le_cate, le_size, le_color, start_year = load_model(tenant_id, 'd')
    
    df['store_id_index'] = le_store.transform(df['store_id'])
    df['category_index'] = le_cate.transform(df['category'])
    df['size_index'] = le_size.transform(df['size'])
    df['color_index'] = le_color.transform(df['color'])
    
    df['month_index'] = (df['year']-start_year)*12 + df['month'] - 1
    
    #Column check
    train_col = ['store_id_index', 'category_index', 'size_index','color_index', 
                 'year', 'month', 'day', 'weekday', 'month_index', 
                 'daily_sku_sales', 'daily_sku_revenue']
    xcol = [c for c in train_col if c not in ['daily_sku_sales', 'daily_sku_revenue']]
    
    X_test = df[xcol]
    
    y_predict = np.round(pred_model.predict(X_test))
    
    #decode
    df_result = df[df_ori_columns]
    df_result['predicted_daily_sales'] = y_predict[:,0]
    df_result['predicted_daily_revenue'] = y_predict[:,1]
    
    return df_result
    
def predict_monthly_sku(tenant_id, df):
    """
    predict
    Args:
        df(dataframe)
    Returns:
        df(dataframe): dataframe of predict result
    """
    #Encode
    df_ori_columns = df.columns
    #Rename
    df=df.rename(index=str, columns={"predict_year": "year", "predict_month": "month"})

    today = datetime.now().date()
    pred_model, le_store, le_cate, le_size, le_color, start_year = load_model(tenant_id, 'm')
    
    df['store_id_index'] = le_store.transform(df['store_id'])
    df['category_index'] = le_cate.transform(df['category'])
    df['size_index'] = le_size.transform(df['size'])
    df['color_index'] = le_color.transform(df['color'])
    
    df['month_index'] = (df['year']-start_year)*12 + df['month'] - 1
    
    #Column check
    train_monthly_col = ['store_id_index', 'category_index', 'size_index','color_index', 
                     'year', 'month', 'month_index', 
                     'monthly_sku_sales', 'monthly_sku_revenue']
    xcol = [c for c in train_monthly_col if c not in ['monthly_sku_sales', 'monthly_sku_revenue']]
    
    X_test = df[xcol]
      
    #pred_model = load_model_monthly(tenant_id, today)
    y_predict = np.round(pred_model.predict(X_test))
    
    # Decode
    df_result = df[df_ori_columns]
    df_result['predicted_monthly_sales'] = y_predict[:,0]
    df_result['predicted_monthly_revenue'] = y_predict[:,1]
    
    return df_result