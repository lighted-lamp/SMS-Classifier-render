from flask import Flask,render_template,request
import pickle

app=Flask(__name__)

import string
import nltk
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


stemmer= SnowballStemmer('english')

def transform(text):
    if not text:  # Covers None or empty string
        return ""
    text= text.lower()
    text= nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)
    text=y[:]
    y=[]
    for i in text:
        y.append(stemmer.stem(i))
    text=' '.join(y)
    return text

tf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_spam():
    msg = request.form.get('message')
    transtext = transform(msg)
    vector = tf.transform([transtext])
    result = model.predict(vector)[0]

    print("Raw prediction from model:", result)  # Add this line

    if result == 1:
        final_result = "Spam Detected!"
    else:
        final_result = "Not Spam"

    return render_template('index.html', result=final_result)

if __name__=='__main__':
    app.run(debug=True)
