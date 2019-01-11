from flask_sqlalchemy import SQLAlchemy, current_app
from sqlalchemy import inspect
import pandas as pd

db = SQLAlchemy()

class Record(db.Model):
    __tablename__ = 'records'
    id = db.Column(db.Integer, nullable=False, primary_key=True, autoincrement=True)

    # original data from csv

    tenant_id = db.Column(db.String(32), nullable=False)
    tenant_name = db.Column(db.String(32))

    transaction_id = db.Column(db.String(32), nullable=False)

    store_id = db.Column(db.String(32), nullable=False)
    store_name = db.Column(db.String(32))

    goods_id = db.Column(db.String(32))
    goods_name = db.Column(db.String(32))

    quantity = db.Column(db.Integer, nullable=False)
    sale_price = db.Column(db.Float, nullable=False)
    tag_price = db.Column(db.Float, nullable=False)

    sale_date = db.Column(db.DateTime, nullable=False, index=True)
    hit_date = db.Column(db.DateTime, nullable=False)
    off_date = db.Column(db.DateTime)
    season = db.Column(db.String(32))

    gender = db.Column(db.String(32))
    category = db.Column(db.String(32), nullable=False)
    style = db.Column(db.String(32))
    size = db.Column(db.String(32), nullable=False)
    color = db.Column(db.String(32), nullable=False)
    material = db.Column(db.String(32))

    inventory = db.Column(db.Integer)
    inventory_store = db.Column(db.Integer)
    inv_to_sales = db.Column(db.Float)
    
    store_rank = db.Column(db.String(32))
    province = db.Column(db.String(32), nullable=False)
    city = db.Column(db.String(32), nullable=False)
    district = db.Column(db.String(32))
    address = db.Column(db.String(32))
    latitude = db.Column(db.String(32))
    longitude = db.Column(db.String(32))

    __table_args__ = (
        db.UniqueConstraint(tenant_id, transaction_id, name='tenant_transaction_idx'),
    )
    
    def __repr__(self):
        return '<Record {}, \ntenant name: {}, \nstore name: {}\n goods name: {}\n>' \
            .format(self.id, self.tenant_name, self.store_name, sefl.goods_name)

def get_record_data_mapping():
    """
    Returns:
        data type(dict): dict of required record data mapping {name, type}
    """
    mapper = inspect(Record)
    required_mapping = {c.name:c.type.python_type for c in mapper.columns if not c.nullable}
    del required_mapping['id']
    mapping = {c.name:c.type.python_type for c in mapper.columns}
    del mapping['id']
    return required_mapping, mapping

required_record_mapping, record_mapping = get_record_data_mapping()

def create_record(record):
    """
    Args:
        record(dict): record dict
    Returns:
        None
    """
    try:
        # row = Record(**record)
        # db.session.add(row)
        db.session.execute(Record.__table__.insert().prefix_with('IGNORE'), record)
        db.session.commit()

    except Exception as error:
        raise ValueError('Error on create record in db: {} \n record: {}'.format(error, record))

def get_records_periods(tenant_id, start, end):
    """
    Query records of a tenant in a period of time [start, end)
    Args:
        tenant_id(string): tenant_id to query
        start(datetime): query start time 
        end(datetime): query end time
    Returns:
        records(dataframe): pandas dataframe of record
    """
    sql_cmd = Record.query.filter_by(tenant_id=tenant_id) \
                          .filter(Record.sale_date < end).filter(Record.sale_date >= start).statement
    records = pd.read_sql(sql = sql_cmd, con = db.session.bind)
    return records

def get_records(tenant_id):
    """
    Query records of a tenant
    Args:
        tenant_id(string): tenant_id to query
    Returns:
        records(dataframe): pandas dataframe of record
    """
    sql_cmd = Record.query.filter_by(tenant_id=tenant_id).statement
    records = pd.read_sql(sql = sql_cmd, con = db.session.bind)
    return records