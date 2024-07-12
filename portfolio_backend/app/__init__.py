from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from config import Config

db = SQLAlchemy()
migrate = Migrate()


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    migrate.init_app(app, db)

    # Enable CORS with specific configuration
    CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

    from app import routes
    app.register_blueprint(routes.bp)

    return app
