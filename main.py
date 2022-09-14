from flask import Flask
from flask import request, jsonify
from moderation import predict

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def do_predict():
    text = request.form['text']
    score = predict(text)
    return jsonify(
        score=int(score)
    )

if __name__ == '__name__':
    app.run('0.0.0.0')