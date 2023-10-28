# Fake News Detection

train.rar(train.csv) is a dataset that contains textual news articles, the goal is to classify the input article as either "Authentic"  or "Fake". We create a pipeline that combines TF-IDF vectorization and LinearSVC.

# Steps to Run

1. python -m virtualenv venv
2. venv\Scripts\activate
    If error: Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser
3. pip install -r requirements.txt
4. Run the cells with "prerequisites.ipynb"
5. In the terminal: streamlit analysis.py (Will take some time to run).

# Reference
- https://www.youtube.com/watch?v=ZE2DANLfBIs
