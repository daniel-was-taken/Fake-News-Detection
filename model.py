import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# Load the data
data = pd.read_csv('data/train.csv')

# Split the data into features (X) and labels (y)
X, y = data['text'], data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline that combines TF-IDF vectorization and LinearSVC
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words="english", max_df=0.7)),
    ('clf', LinearSVC(dual=True))
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Evaluate the model
accuracy = pipeline.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save the trained pipeline (including vectorizer and classifier) using pickle
with open('models/final_pipeline.pickle', 'wb') as pipeline_file:
    pickle.dump(pipeline, pipeline_file)

# Load the trained pipeline
news_detection = pickle.load(open('models/final_pipeline.pickle', 'rb'))
