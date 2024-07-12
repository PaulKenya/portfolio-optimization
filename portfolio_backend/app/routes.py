from flask import Blueprint, request, jsonify
from .models import Request, db
import re
import uuid

bp = Blueprint('main', __name__)

INTERVAL_REGEX = re.compile(r"(\d+)([smhdMy])")
LOOK_BACK_PERIOD_REGEX = re.compile(r"(\d+)([smhdMy])")


@bp.route('/requests', methods=['POST'])
def create_request():
    data = request.get_json()
    if not INTERVAL_REGEX.match(data.get('interval', '')) or not LOOK_BACK_PERIOD_REGEX.match(data.get('look_back_period', '')):
        return jsonify({'error': 'Invalid format for interval or look_back_period'}), 400

    new_request = Request(
        assets=data.get('assets', []),
        interval=data.get('interval'),
        look_back_period=data.get('look_back_period'),
        investment_amount=data.get('investment_amount')
    )
    db.session.add(new_request)
    db.session.commit()
    return jsonify(new_request.serialize()), 201


@bp.route('/requests/<uuid:id>', methods=['GET'])
def get_request(id):
    req = Request.query.get_or_404(id.bytes)
    return jsonify(req.serialize())


@bp.route('/requests/<uuid:id>', methods=['PUT'])
def update_request(id):
    data = request.get_json()
    req = Request.query.get_or_404(id.bytes)

    if not INTERVAL_REGEX.match(data.get('interval', req.interval)) or not LOOK_BACK_PERIOD_REGEX.match(data.get('look_back_period', req.look_back_period)):
        return jsonify({'error': 'Invalid format for interval or look_back_period'}), 400

    req.assets = data.get('assets', req.assets)
    req.interval = data.get('interval', req.interval)
    req.look_back_period = data.get('look_back_period', req.look_back_period)
    req.investment_amount = data.get('investment_amount', req.investment_amount)

    db.session.commit()
    return jsonify(req.serialize()), 200


@bp.route('/requests/<uuid:id>', methods=['DELETE'])
def delete_request(id):
    req = Request.query.get_or_404(id.bytes)
    db.session.delete(req)
    db.session.commit()
    requests = Request.query.all()
    return jsonify([req.serialize() for req in requests]), 200


@bp.route('/requests', methods=['GET'])
def get_requests():
    requests = Request.query.all()
    return jsonify([req.serialize() for req in requests])
