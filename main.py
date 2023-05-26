from flask import Flask, request, jsonify

from internal.moderation import comprehensive_checks, moderation
from pkg.logger.logger import logger
from pkg.config.config import cfg


API_KEY = cfg.get("API_KEY")

app = Flask(__name__)

@app.route("/text-moderation/health", methods=["GET"])
def health():
    return jsonify({"status": 200, "data": "Hello World"})

@app.route("/text-moderation/checks", methods=["POST"])
def checks():
    api_key = request.headers.get("x-api-key")
    if api_key is None or api_key != API_KEY:
        return jsonify({"status": 401, "data": "Unauthorized"})

    try:
        text = request.get_json()['text']
        return jsonify({"status": 200, "data": comprehensive_checks(text)})
    except Exception as exc:
        logger.exception(exc)
        return jsonify({"status": 500, "data": exc})

@app.route("/text-moderation/predict", methods=['POST'])
def predict():
    api_key = request.headers.get("x-api-key")
    if api_key is None or api_key != API_KEY:
        return jsonify({"status": 401, "data": "Unauthorized"})

    try:
        text = request.get_json()['text']
        return jsonify({"status": 200, "data": moderation(text)})
    except Exception as exc:
        logger.exception(exc)
        return jsonify({"status": 500, "data": exc})


if __name__ == '__name__':
    port = int(cfg.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
