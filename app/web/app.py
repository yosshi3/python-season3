from flask import Flask,send_from_directory

import sys,os
sys.path.append(os.path.abspath(".."))
from predictor import Predictor

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, ''), 'favicon.ico')

@app.route("/<req>")
def index(req):
    pre = Predictor('../')
    return pre.predict(req)

if __name__ == "__main__":
    app.run(port=8888)
