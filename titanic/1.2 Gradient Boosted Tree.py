# Databricks notebook source
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType, IntegerType

# COMMAND ----------

training = spark.table("vida_kaggle.titanic_train")

# COMMAND ----------

# MAGIC %sql select sum(int(Age)) / count(Age) from vida_kaggle.titanic_train where age is not null 

# COMMAND ----------

training = training.withColumn("Pclass2", training["Pclass"].cast(IntegerType()))
training = training.withColumn("Age2", training["Age"].cast(DoubleType()))
training = training.withColumn("SibSp2", training["SibSp"].cast(IntegerType()))
training = training.withColumn("Parch2", training["Parch"].cast(IntegerType()))
training = training.withColumn("Fare2", training["Fare"].cast(DoubleType()))
training = training.fillna(-1, subset=["Pclass2", "SibSp2", "Parch2", "Fare2"])
training =  training.fillna(29.67, subset=["Age2"])

# COMMAND ----------

testing = spark.table("vida_kaggle.titanic_test")
testing = testing.withColumn("Pclass2", testing["Pclass"].cast(IntegerType()))
testing = testing.withColumn("Age2", testing["Age"].cast(DoubleType()))
testing = testing.withColumn("SibSp2", testing["SibSp"].cast(IntegerType()))
testing = testing.withColumn("Parch2", testing["Parch"].cast(IntegerType()))
testing = testing.withColumn("Fare2", testing["Fare"].cast(DoubleType()))
testing = testing.fillna(-1, subset=["Pclass2", "SibSp2", "Parch2", "Fare2"])
testing = testing.fillna(29.67, subset=["Age2"])

# COMMAND ----------

display(training)

# COMMAND ----------

labelIndexer = StringIndexer(inputCol="Survived", outputCol="indexedLabel").fit(training)

# COMMAND ----------

trainingFeatureTest = labelIndexer.transform(training)
display(trainingFeatureTest.select("Survived", "indexedLabel"))

# COMMAND ----------

featureIndexer1 = StringIndexer(inputCol="Sex", outputCol="feature1").fit(training)

# COMMAND ----------

trainingFeatureTest = featureIndexer1.transform(trainingFeatureTest)
display(trainingFeatureTest.select("Survived", "indexedLabel", "Sex", "feature1"))

# COMMAND ----------

featureIndexer2 = StringIndexer(inputCol="Embarked", outputCol="feature2").setHandleInvalid("keep").fit(training)

# COMMAND ----------

trainingFeatureTest = featureIndexer2.transform(trainingFeatureTest)
display(trainingFeatureTest.select("Survived", "indexedLabel", "Embarked", "feature2"))

# COMMAND ----------

display(trainingFeatureTest.select("indexedLabel", "feature1", "feature2", "Pclass2", "Age2", "SibSp2", "Parch2", "Fare2"))

# COMMAND ----------

featureAssembler = VectorAssembler(
    inputCols=["feature1", "feature2", "Age2","Pclass2", "SibSp2", "Parch2", "Fare2"],
    outputCol="features")

# COMMAND ----------

trainingFeatureTest = featureAssembler.transform(trainingFeatureTest)
display(trainingFeatureTest.select("Survived", "indexedLabel", "Embarked", "feature2", "features"))

# COMMAND ----------

# Train a GBT model.
gbtClassifier = GBTClassifier(labelCol="indexedLabel", featuresCol="features", maxIter=10)

# COMMAND ----------

pipeline = Pipeline(stages=[labelIndexer, featureIndexer1, featureIndexer2, featureAssembler, gbtClassifier])

# COMMAND ----------

model = pipeline.fit(training)

# COMMAND ----------

treeModel = model.stages[-1]
# summary only
#display(treeModel)

# COMMAND ----------

training_predictions = model.transform(training)

# COMMAND ----------

display(training_predictions.select("indexedLabel", "prediction"))

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(training_predictions)
print("Test Error = %g " % (1.0 - accuracy))

# COMMAND ----------

predictions = model.transform(testing)

# COMMAND ----------

predictions.count()

# COMMAND ----------

output = predictions.withColumn("Survived", predictions["prediction"].cast(IntegerType()))

# COMMAND ----------

display(output.select("PassengerId", "Survived"))

# COMMAND ----------

# MAGIC %fs rm -r /FileStore/tables/vida/kaggle/titanic/outputtest

# COMMAND ----------

(output
  .select("PassengerId","Survived")
  .repartition(1)
  .write
  .format("com.databricks.spark.csv")
  .option("header", "true")
  .save("/FileStore/tables/vida/kaggle/titanic/outputtest"))

# COMMAND ----------

displayHTML("<a href=\"files/tables/vida/kaggle/titanic/outputtest/part-00000-tid-1840240035864298791-aa011da6-b83a-471d-a049-c6ae0622f983-801-c000.csv\">Download Here</a>")

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/vida/kaggle/titanic/outputtest

# COMMAND ----------

predictions.registerTempTable("predictions")

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT
# MAGIC   Sex, Pclass, Embarked, SibSp, prediction, count(*) 
# MAGIC FROM
# MAGIC   predictions 
# MAGIC GROUP BY 
# MAGIC   Sex, Pclass, Embarked, SibSp, prediction 
# MAGIC ORDER BY 
# MAGIC   Sex, Pclass, Embarked, SibSp

# COMMAND ----------

display(predictions.select("Sex", "Pclass", "prediction"))

# COMMAND ----------

