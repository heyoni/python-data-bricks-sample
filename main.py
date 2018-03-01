import csv
import os

import pyspark
from pgpy import PGPKey, PGPMessage, constants
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import percent_rank, col
from pyspark.sql.window import Window

os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3.6'
os.environ['PYSPARK_PYTHON'] = 'python3.6'

# Decrypt CSV file
# https://pgpy.readthedocs.io/en/latest/examples.html
titanic = open('titanic.csv', 'w')
key, _ = PGPKey.from_file('slim.shady.sec.asc')

message_from_blob = PGPMessage.from_file('titanic.csv.gpg')
dec_mess = key.decrypt(message_from_blob)
csv_result = dec_mess.message
titanic.write(csv_result.decode('UTF-8'))

# Setting up our DataFrame
sc = pyspark.SparkContext()
sql_ctx = pyspark.SQLContext(sc)

# Explicitly setting the schema type - may be unnecessary
# Notes: https://stackoverflow.com/questions/44706398/spark-csv-reader-quoted-numerics
schema = StructType([
    StructField("RecId", IntegerType(), True),
    StructField("Name", StringType(), True),
    StructField("PassengerClass", StringType(), True),
    StructField("Age", IntegerType(), True),
    StructField("Sex", StringType(), True),
    StructField("Survived", IntegerType(), True),
    StructField("SexCode", IntegerType(), True),
])

# Print the average age
titanic = sql_ctx.read.option("delimiter", ",").option("header", True).schema(schema).csv('titanic.csv')
titanic.groupBy().avg('Age').show()

# Print the top 75th percentile in age
# Notes: https://databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html
# https://stackoverflow.com/questions/40885266/how-to-take-the-top-percentage-from-a-relatively-large-spark-dataframe-and-save

window = Window.partitionBy().orderBy(titanic['Age'].asc())
titanic.select('*', percent_rank().over(window).alias('age_rank')).filter(col('age_rank') >= 0.75).show(1000)

# Saving our DF to Parquet file
titanic.write.parquet('titanic_p', mode='overwrite')
