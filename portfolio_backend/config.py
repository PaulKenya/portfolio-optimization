import os


class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URI', 'mysql+pymysql://root:@localhost/portfolio')
    SQLALCHEMY_TRACK_MODIFICATIONS = False