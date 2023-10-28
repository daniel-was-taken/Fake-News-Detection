import os
import findspark
from pyspark.sql import SparkSession
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pickle
from sparkinit import spark
from sklearn.metrics import confusion_matrix
from model import X_test, y_test
import seaborn as sns

df1 = spark.read.csv('data/train.csv', header=True, inferSchema=True)


clf = pickle.load(open('models/final_pipeline.pickle', 'rb'))
# Streamlit code starts here
st.title("Fake News Detection")

data = st.text_area(label='Enter News')
prediction = clf.predict([data])
result_message = 'Authentic' if prediction == 1 else 'Fake'

st.subheader(f"The News is {result_message}")

st.title("Dataset Analysis")

# Display the first 5 rows of the DataFrame
st.subheader("Displaying the first 5 rows of the DataFrame:")
st.dataframe(df1.limit(5))

# Show summary statistics
st.subheader("Summary Statistics:")
st.write(df1.describe().toPandas())

# Show specific columns
selected_columns = st.multiselect("Select columns to display:", df1.columns)
if selected_columns:
    st.dataframe(df1.select(selected_columns).limit(10))

# Show a count of rows
st.subheader("Row Count:")
st.write(f"The DataFrame contains {df1.count()} rows.")


st.title("Pie Chart")

lf = df1[df1["label"] == 0].count()
lt = df1[df1["label"] == 1].count()

labels = ['Authentic', 'Fake']
sizes = [lt, lf]

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.

# # Display the pie chart using Streamlit
st.pyplot(fig)

# # Optional: Display the count of Authentic and Fake labels
st.subheader("Label Counts")
st.write("Authentic:", lt)
st.write("Fake:", lf)


st.title("Bar Chart")

fig, ax = plt.subplots()
ax.bar(labels, sizes)

# Add labels and title
ax.set_xlabel('Category')
ax.set_ylabel('Count')
ax.set_title('Authentic vs. Fake Label Counts')

# Display the bar graph using Streamlit
st.pyplot(fig)

# Confusion matrix
st.title("Confusion Matrix")
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6, 4))
# plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5, linecolor='black')
st.pyplot(plt)