import os
import findspark
from pyspark.sql import SparkSession

java_home_path = os.environ.get("JAVA_HOME")

findspark.init()
#os.environ["JAVA_HOME"] = java_home_path
os.environ["SPARK_HOME"] = findspark.find()

spark = SparkSession.builder \
    .master("local") \
    .appName("Colab") \
    .config('spark.ui.port', '4051') \
    .getOrCreate()

