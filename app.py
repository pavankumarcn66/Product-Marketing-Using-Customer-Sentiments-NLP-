from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np
import re

app = Flask(__name__)

model = load_model('FINAL_BI_GRU_output.h5')


@app.route('/')
def home():
    return render_template('home.html')


@app.route("/predict", methods=['POST'])
def predict():
    message = request.form["message"]

    message = [message]

    def clean_text(data):
        data = re.sub(r'http\S+', '', data)
        data = re.sub('[^a-zA-Z]', ' ', data)
        data = data.lower()

        return data

    texts = [''.join(clean_text(text)) for text in message]


    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(texts)


    # seq = tokenizer.texts_to_sequences(message)
    # padded = pad_sequences(seq, maxlen=68)

    # voc_size = 70
    onehot_repr_short = [one_hot(words, 68) for words in texts]

    # sent_length = 500
    embedded_docs_msg = pad_sequences(onehot_repr_short, padding='pre', maxlen=68)

    pred = model.predict(embedded_docs_msg)

    # pred = model.predict(padded)
    class_names = ['Neutral', 'Negative', 'Positive']
    my_pred = class_names[np.argmax(pred)]

    return render_template('Article.html', data=my_pred)


if __name__ == "__main__":
    app.run(debug=True)
