# Databricks notebook source
from pyspark.ml import Pipeline

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

from pyspark.sql.types import DoubleType, IntegerType

# COMMAND ----------

training = spark.table("vida_kaggle.housing_prices_train")

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Make sure there are no null or negative values in these columns.

# COMMAND ----------

# MAGIC %sql select BedroomAbvGr from vida_kaggle.housing_prices_train where BedroomAbvGr is null or BedroomAbvGr < 0

# COMMAND ----------

# MAGIC %sql select 1stFlrSF from vida_kaggle.housing_prices_train where 1stFlrSF is null or BedroomAbvGr < 0

# COMMAND ----------

# MAGIC %sql select 2ndFlrSF from vida_kaggle.housing_prices_train where 2ndFlrSF is null or BedroomAbvGr < 0

# COMMAND ----------

# MAGIC %sql select 2ndFlrSF from vida_kaggle.housing_prices_train where 2ndFlrSF = 0 limit 2

# COMMAND ----------

training = training.withColumn("feature1", training["BedroomAbvGr"].cast(IntegerType()))
training = training.withColumn("feature2", training["1stFlrSF"] + training["2ndFlrSF"])
training = training.withColumn("label", training["SalePrice"].cast(DoubleType()))

# COMMAND ----------

testing = spark.table("vida_kaggle.housing_prices_test")
testing = testing.withColumn("feature1", testing["BedroomAbvGr"].cast(IntegerType()))
testing = testing.withColumn("feature2", testing["1stFlrSF"] + testing["2ndFlrSF"])

# COMMAND ----------

display(testing)

# COMMAND ----------

featureAssembler = VectorAssembler(
    inputCols=["feature1", "feature2"],
    outputCol="features")

# COMMAND ----------

# Train a Linear Regression model.
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# COMMAND ----------

pipeline = Pipeline(stages=[featureAssembler, lr])

# COMMAND ----------

model = pipeline.fit(training)

# COMMAND ----------

lr_model = model.stages[-1]

# COMMAND ----------

print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

# COMMAND ----------

trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

# COMMAND ----------

training_predictions = model.transform(training)

# COMMAND ----------

display(training_predictions.select("label", "prediction"))

# COMMAND ----------

predictions = model.transform(testing)

# COMMAND ----------

predictions.count()

# COMMAND ----------

output = predictions.withColumn("SalePrice", predictions["prediction"].cast(DoubleType()))

# COMMAND ----------

display(output.select("Id", "SalePrice"))

# COMMAND ----------

# MAGIC %fs rm -r /FileStore/tables/vida/kaggle/housing-prices/outputtest

# COMMAND ----------

(output
  .select("Id","SalePrice")
  .repartition(1)
  .write
  .format("com.databricks.spark.csv")
  .option("header", "true")
  .save("/FileStore/tables/vida/kaggle/housing-prices/outputtest"))

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/vida/kaggle/housing-prices/outputtest

# COMMAND ----------

displayHTML("<a href=\"files/tables/vida/kaggle/housing-prices/outputtest/part-00000-tid-3510919909891879050-d9aa4fb2-471c-4c87-8b0e-297c404c6259-68-1-c000.csv\">Download Here</a>")