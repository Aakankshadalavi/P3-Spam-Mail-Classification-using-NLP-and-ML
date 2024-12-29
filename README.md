<h1>P3-Spam-Mail-Classification-using-NLP-and-ML</h1> 
<p>The Spam Mail Classification project aims to automatically detect and classify emails as either "spam" or "ham" (non-spam). By leveraging Natural Language Processing (NLP) and Machine Learning (ML) techniques, this project helps identify unwanted emails, improving the efficiency of email systems by filtering out unnecessary or potentially harmful messages.
The project focuses on training a machine learning model to recognize patterns and features in the content of emails, such as the presence of certain words or phrases commonly found in spam emails. With the help of NLP, the text in emails is processed, cleaned, and transformed into a format that a machine learning model can understand. After training, the model can predict whether an email is spam or not.</p>
<h2>Steps to Run the Project:</h2>
<p>Here’s a simple guide to running the Spam Mail Classification project:</p>

<h3>1.Install Required Libraries:</h3> <p>Before starting, make sure you have the necessary libraries installed. You can do this using the following Python commands:</p>
<p>pip install pandas numpy scikit-learn nltk </p>
<h3>2.Prepare the Dataset:</h3> <p>The first step is to obtain a dataset of labeled emails. A popular dataset for spam classification is the SMS Spam Collection Dataset or Enron Spam Dataset. This dataset consists of emails, each labeled as spam or ham (non-spam).</p>
<h3>3.Load the Dataset:</h3> <p>After downloading the dataset, load it into Python using Pandas:</p>
<p>import pandas as pd
data = pd.read_csv("spam_data.csv")
</p>
<h3>4.Preprocess the Data: </h3><p>This step involves cleaning and preparing the text data for analysis. Common preprocessing steps include:</p>
<p>
  Converting all text to lowercase
  Removing punctuation, stopwords, and numbers
  Tokenizing the text into words
  Lemmatizing or stemming the words</p>
  <p>Example code for text preprocessing using NLTK:</p>
  <p>import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return " ".join(tokens)

data['processed_text'] = data['text'].apply(preprocess)
</p>
<h3>5.Feature Extraction: </h3>
<p>Use TF-IDF (Term Frequency-Inverse Document Frequency) to convert the text into numerical features that can be fed into the machine learning model.
</p>
<p>
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['processed_text'])
y = data['label']  # Assuming the label column is 'spam' or 'ham'
  </p>  
  <h3>6.Train a Machine Learning Model</h3>
  <p>Use any machine learning algorithm like Logistic Regression, Naive Bayes, or SVM for classification. Here's how you can train a model using Naive Bayes:
</p>
<p>
  from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# Predicting the labels on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
</p>
<h3>7.Evaluate the Model:</h3><p>After training the model, evaluate its performance using metrics such as accuracy, precision, recall, and F1-score to understand how well the model is detecting spam emails.

Example:</p>
<p>
  from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
</p>
<h3>8.Make Predictions:</h3><p>Once the model is trained and evaluated, you can use it to predict whether a new email is spam or ham. For example:</p>
<p>
  new_email = "Congratulations! You've won a lottery. Click here to claim your prize!"
processed_email = preprocess(new_email)
email_features = vectorizer.transform([processed_email])
prediction = model.predict(email_features)

if prediction == 'spam':
    print("The email is spam.")
else:
    print("The email is ham.")
</p>
<p>By following these steps, you can build and run a spam mail classification system using NLP and machine learning techniques.</p>
