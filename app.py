import tensorflow as tf

from flask import Flask, render_template, request, url_for, flash, redirect

tfmodel = tf.saved_model.load("transformer_batch_size-64__num_layers-4__d_model-128__dff-256__num_heads-8__dropout-0.3__epochs-200_")




app = Flask(__name__)

translations = []

@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        content = request.form["content"]

    translations.append(
        {"input": content,
         "output": tfmodel(tf.constant(content.decode("utf-8"))).decode("utf-8")})

    return render_template("index.html", translations=translation)
