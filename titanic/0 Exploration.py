# Databricks notebook source
# MAGIC %sql create database if not exists vida_kaggle

# COMMAND ----------

# MAGIC %sql use vida_kaggle

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/vida/kaggle

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/vida/kaggle/titanic

# COMMAND ----------

# MAGIC %sql CREATE TABLE if not exists titanic_train
# MAGIC   USING csv
# MAGIC   OPTIONS (path "dbfs:/FileStore/tables/vida/kaggle/titanic/train.csv", header "true", mode "FAILFAST")

# COMMAND ----------

# MAGIC %sql CREATE TABLE if not exists titanic_test
# MAGIC   USING csv
# MAGIC   OPTIONS (path "dbfs:/FileStore/tables/vida/kaggle/titanic/test.csv", header "true", mode "FAILFAST")

# COMMAND ----------

# MAGIC %sql show tables

# COMMAND ----------

# MAGIC %sql select * from titanic_train

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Data Dictionary
# MAGIC 
# MAGIC | Variable | Definition | Key |
# MAGIC | --------- | -----| -----|
# MAGIC |survival	| Survival	| 0 = No, 1 = Yes |
# MAGIC |pclass	| Ticket class	| 1 = 1st, 2 = 2nd, 3 = 3rd |
# MAGIC |sex	| Sex | |
# MAGIC |Age	| Age in years	| |
# MAGIC |sibsp	| # of siblings / spouses aboard the Titanic	| |
# MAGIC |parch	| # of parents / children aboard the Titanic	| |
# MAGIC |ticket	| Ticket number	| |
# MAGIC |fare	| Passenger fare | |
# MAGIC |cabin	| Cabin number	| |
# MAGIC |embarked	| Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton  |  |
# MAGIC 
# MAGIC Variable Notes
# MAGIC 
# MAGIC pclass: A proxy for socio-economic status (SES)
# MAGIC 1st = Upper
# MAGIC 2nd = Middle
# MAGIC 3rd = Lower
# MAGIC 
# MAGIC age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# MAGIC 
# MAGIC sibsp: The dataset defines family relations in this way...
# MAGIC Sibling = brother, sister, stepbrother, stepsister
# MAGIC Spouse = husband, wife (mistresses and fiancés were ignored)
# MAGIC 
# MAGIC parch: The dataset defines family relations in this way...
# MAGIC Parent = mother, father
# MAGIC Child = daughter, son, stepdaughter, stepson
# MAGIC Some children travelled only with a nanny, therefore parch=0 for them.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC You were more likely to survive the Titantic diaster if you were of a higher ticket class

# COMMAND ----------

# MAGIC %sql select Pclass, Survived, count(*) from titanic_train group by PClass, Survived order by Pclass, Survived asc

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC You were more likely to survive the Titanic if you were a woman.

# COMMAND ----------

# MAGIC %sql select sex, Survived, count(*) from titanic_train group by sex, Survived order by sex, Survived asc

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Plot of age vs. survival rate.  Probably need to classify as child or not.

# COMMAND ----------

# MAGIC %sql select cast(age as decimal) as age_numerical, survived, count(*) from titanic_train group by age_numerical, survived

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Fare vs. survival.

# COMMAND ----------

# MAGIC %sql select cast(fare as decimal) as fare_numerical, survived, count(*) from titanic_train group by fare_numerical, survived

# COMMAND ----------

# MAGIC %md The C embarkment folks were more likely to survive than other embarkment areas.

# COMMAND ----------

# MAGIC %sql select embarked, Survived, count(*) from titanic_train group by embarked, Survived order by Survived, embarked asc

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Being in certain cabins was better than not having a cabin listed.

# COMMAND ----------

# MAGIC %sql select substr(cabin, 0, 1), Survived, count(*) from titanic_train group by substr(cabin, 0, 1), Survived order by substr(cabin, 0, 1), Survived

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Those with one sibling seemed more likely to survive than those with none.

# COMMAND ----------

# MAGIC %sql select sibsp, Survived, count(*) from titanic_train group by sibsp, Survived order by sibsp, Survived asc

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC People who had some family onboard the Titanic were more likely to survive than those without.

# COMMAND ----------

# MAGIC %sql select parch, Survived, count(*) from titanic_train group by parch, Survived order by parch, Survived asc

# COMMAND ----------

