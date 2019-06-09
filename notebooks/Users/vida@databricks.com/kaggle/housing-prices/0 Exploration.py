# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Housing Prices Exploration

# COMMAND ----------

# MAGIC %sql create database if not exists vida_kaggle

# COMMAND ----------

# MAGIC %sql use vida_kaggle

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC See data_description.txt for more details on the columns for this dataset.

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/vida/kaggle/housing-prices

# COMMAND ----------

# MAGIC %sh tail -20 /dbfs/FileStore/tables/vida/kaggle/housing-prices/data_description.txt

# COMMAND ----------

# MAGIC %sql CREATE TABLE if not exists housing_prices_train
# MAGIC   USING csv
# MAGIC   OPTIONS (path "dbfs:/FileStore/tables/vida/kaggle/housing-prices/train.csv", header "true", mode "FAILFAST")

# COMMAND ----------

# MAGIC %sql select * from housing_prices_train

# COMMAND ----------

spark.table("housing_prices_train").printSchema()

# COMMAND ----------

# MAGIC %sql CREATE TABLE if not exists housing_prices_test
# MAGIC   USING csv
# MAGIC   OPTIONS (path "dbfs:/FileStore/tables/vida/kaggle/housing-prices/test.csv", header "true", mode "FAILFAST")

# COMMAND ----------

# MAGIC %sql select * from housing_prices_test

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC More Bedrooms seems a bit correlated with a higher sales price.

# COMMAND ----------

# MAGIC %sql select int(BedroomAbvGr), double(SalePrice) from housing_prices_train

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Total Sq Foot vs. sales price.

# COMMAND ----------

# MAGIC %sql select int(1stFlrSF) + int(2ndFlrSF) as BothFlrSF, double(SalePrice) from housing_prices_train

# COMMAND ----------

