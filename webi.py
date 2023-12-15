import streamlit as st
import pickle
import nltk
nltk.download("punkt")
nltk.download("stopwords")
import sklearn


from nltk.corpus import stopwords
stopwords.words("english")

import string
string_punc = string.punctuation

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
ps.stem("swimming")

def transform_text(text):
  text=text.lower()

  text=nltk.word_tokenize(text)

  y=[]
  for i in text:
    if i.isalnum():
       y.append(i)

  text=y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words("english") and i not in string_punc :
      y.append(i)

  text=y[:]
  y.clear()
  
  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

tfidf = pickle.load(open("C:\\Users\\Hp\\.vscode\\extensions\\python_project2\\vectorizer.pkl","rb"))

model = pickle.load(open("C:\\Users\\Hp\\.vscode\\extensions\\python_project2\\model.pkl","rb"))

st.title("Email/SMS Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
# preprocessing data

  transformed_sms = transform_text(input_sms)

#vectorization

  vector_input = tfidf.transform([transformed_sms])

# predict

  result = model.predict(vector_input)[0]

# display

  if result == 1:
    st.header("Spam")

  else:
    st.header("Not Spam") 
 




