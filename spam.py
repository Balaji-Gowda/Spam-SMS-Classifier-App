from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.corpus import stopwords
import re
import nltk
import pickle
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
mod_pick = pickle.load(open('model.pkl', 'rb'))
trans_pick = pickle.load(open('transform.pkl', 'rb'))

spam = Flask(__name__)


#########################################################################################################
ss = SnowballStemmer('english')
stp = stopwords.words('english')


def cleaning(strr):
    txt = re.sub('[^a-zA-Z]', ' ', strr)
    txt = txt.lower()
    txt = txt.split()
    words = [ss.stem(word) for word in txt if word not in stp]
    return " ".join(words)


'''
df = pd.read_csv("SMSSpamCollection.txt", sep='\t',
                 names=["Labels", 'Message'])

# cleaning data

# nltk.download("stopwords")
ps = PorterStemmer()
corpus = []
for i in range(0, len(df)):
    rev = re.sub('[^a-zA-Z]', ' ', df['Message'][i])
    rev = rev.lower()
    rev = rev.split()

    rev = [ps.stem(word)
           for word in rev if not word in set(stopwords.words('english'))]
    rev = ' '.join(rev)
    corpus.append(rev)

cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(df['Labels'])
y = y.iloc[:, 1].values

# train test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

nb = MultinomialNB()
mod = nb.fit(x_train, y_train)

pred = mod.predict(x_test)
con_mat = confusion_matrix(y_test, pred)
score = accuracy_score(y_test, pred)
'''
######################################################################################################


@spam.route('/')
def index():
    return render_template("index.html")


@spam.route('/results/<val>')
def results(val):
    val = val
    return render_template('result.html', prd=val)


@spam.route('/pred', methods=['POST', 'GET'])
def predict():
    msg = ""
    prd = 23
    prd1=''
    op_x = 7
    res = ''
    name = ''
    if request.method == "POST":
        msg = request.form['sms']
        name = request.form['name']
        op = cleaning(msg)
        lst = [op]
        op_x = trans_pick.transform(lst).toarray()
        pred_x = mod_pick.predict(op_x)
        prd = pred_x[0]
        # pred_x = op_x.shape
        if prd == 0:
            res = 'SPAM'
        else:
            res = 'HAM(Not a SPAM)'
        prd1 = 'Hello '+name+' Your Message is: '+res

    # return redirect(url_for('results', val='Hello '+name+' Your Message is: '+res))
    return render_template('index.html',  prd=prd1, res=prd)


if __name__ == "__main__":
    spam.run(debug=True)
