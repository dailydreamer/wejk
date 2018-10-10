from api.db import db
from api.index import create_app

app = create_app()
db.drop_all(app=app)
db.create_all(app=app)
