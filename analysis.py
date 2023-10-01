import os
import findspark
from pyspark.sql import SparkSession
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pickle
from sparkinit import spark
from wordcloud import WordCloud

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



# # Plot a histogram
# st.subheader("Histogram:")
# selected_column_hist = st.selectbox("Select a column for the histogram:", df1.columns)
# if selected_column_hist:
#     hist_data = df1.select(selected_column_hist).toPandas()
#     plt.figure(figsize=(8, 6))
#     plt.hist(hist_data[selected_column_hist], bins=20, edgecolor='k')
#     plt.xlabel(selected_column_hist)
#     plt.ylabel("Frequency")
#     st.pyplot(plt)

# # Scatter plot
# st.subheader("Scatter Plot:")
# x_axis = st.selectbox("Select the x-axis column:", df1.columns)
# y_axis = st.selectbox("Select the y-axis column:", df1.columns)
# if x_axis and y_axis:
#     scatter_data = df1.select(x_axis, y_axis).toPandas()
#     fig = px.scatter(scatter_data, x=x_axis, y=y_axis, title="Scatter Plot")
#     st.plotly_chart(fig)

# Streamlit code ends here

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
