from flask import Flask, render_template

app = Flask(__name__)

messages = [{"input" : "test_input",
             "output": "test_output"
             }]

@app.route("/")
def index():
    return render_template("index.html", messages=messages)
