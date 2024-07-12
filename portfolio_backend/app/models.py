import uuid
from . import db
from sqlalchemy.dialects.mysql import BINARY, VARCHAR, FLOAT
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.types import JSON


class Request(db.Model):
    id = db.Column(BINARY(16), primary_key=True, default=lambda: uuid.uuid4().bytes)
    assets = db.Column(MutableList.as_mutable(JSON))
    interval = db.Column(VARCHAR(10), nullable=False)
    look_back_period = db.Column(VARCHAR(10), nullable=False)
    investment_amount = db.Column(FLOAT, nullable=False)

    def serialize(self):
        return {
            'id': str(uuid.UUID(bytes=self.id)),
            'assets': self.assets,
            'interval': self.interval,
            'look_back_period': self.look_back_period,
            'investment_amount': self.investment_amount
        }
