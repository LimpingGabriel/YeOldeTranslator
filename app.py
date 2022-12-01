import tensorflow_text
import tensorflow as tf

from flask import Flask, render_template, request, url_for, flash, redirect

tfmodel = tf.saved_model.load(
    "C:\\Users\\Benjamin\\OneDrive\\University\\Undergraduate\\MAIS 202\\YeOldeTranslator\\transformer_batch_size-64__num_layers-4__d_model-128__dff-256__num_heads-8__dropout-0.3__epochs-200_",
    )

def get_translation(message):
    return tfmodel(tf.constant(message)).numpy().decode("utf-8")

app = Flask(__name__)

translations = []


@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        content = request.form["content"]

        translated_content = get_translation(content)
        translations.append(
            {"input": content,
             "output": translated_content})

    return render_template("index.html", translations=translations)
