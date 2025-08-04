---
lab:
  title: "Use Delta Lake in Azure Databricks"
---

# Use Delta Lake in Azure Databricks

Delta Lake is an open source project to build a transactional data storage layer for Spark on top of a data lake. Delta Lake adds support for relational semantics for both batch and streaming data operations, and enables the creation of a _Lakehouse_ architecture in which Apache Spark can be used to process and query data in tables that are based on underlying files in the data lake.

This lab will take approximately **30** minutes to complete.

## Create a notebook and ingest data

Now let's create a Spark notebook and import the data that we'll work with in this exercise.

1. In the sidebar, use the **(+) New** link to create a **Notebook**.

1. Change the default notebook name (**Untitled Notebook _[date]_**) to `Explore Delta Lake` and in the **Connect** drop-down list, select your cluster if it is not already selected. If the cluster is not running, it may take a minute or so to start.

1. In the first cell of the notebook, enter the following code, which uses _shell_ commands to download data files from GitHub into the file system used by your cluster.

   ```python
   # Create the folder in FileStore to upload products.csv
   dbutils.fs.mkdirs("/FileStore/delta_lab/")
   dbutils.fs.cp("https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/products.csv", "/FileStore/delta_lab/products.csv")
   ```

1. Use the **&#9656; Run Cell** menu option at the left of the cell to run it. Then wait for the Spark job run by the code to complete.

1. Under the existing code cell, use the **+ Code** icon to add a new code cell. Then in the new cell, enter and run the following code to load the data from the file and view the first 10 rows.

   ```python
   df = spark.read.load('/FileStore/delta_lab/products.csv', format='csv', header='true',inferSchema='true')
   display(df.limit(10))
   ```

## Load the file data into a delta table

The data has been loaded into a dataframe. Let's persist it into a delta table.

1. Add a new code cell and use it to run the following code:

   ```python
   delta_table_path = '/FileStore/delta/products_delta'
   df.write.format('delta').save(delta_table_path)
   ```

   The data for a delta lake table is stored in Parquet format. A log file is also created to track modifications made to the data.

1. Add a new code cell and use it to run the following shell command to view the contents of the folder where the delta data has been saved.

   ```
   dbutils.fs.ls(delta_table_path)
   ```

1. The file data in Delta format can be loaded into a **DeltaTable** object, which you can use to view and update the data in the table. Run the following code in a new cell to update the data; reducing the price of product 771 by 10%.

   ```python
   from delta.tables import *
   from pyspark.sql.functions import *

   # Create a deltaTable object
   deltaTable = DeltaTable.forPath(spark, delta_table_path)
   # update the table (reduce the price of product 771 by     10%)
   deltaTable.update(
   condition = col("ProductID") == 771,
   set = { "ListPrice": col("ListPrice") * 0.9 }
   )
   # View the updated data as a dataframe
   deltaTable.toDF().show(10)
   ```

   The update is persisted to the data in the delta folder, and will be reflected in any new dataframe loaded from that location.

1. Run the following code to create a new dataframe from the delta table data:

   ```python
   new_df = spark.read.format('delta').load(delta_table_path)
   new_df.filter(col("ProductID") == 771).show()
   ```

## Explore logging and _time-travel_

Data modifications are logged, enabling you to use the _time-travel_ capabilities of Delta Lake to view previous versions of the data.

1. In a new code cell, use the following code to view the original version of the product data:

   ```python
   new_df = spark.read.format("delta").option("versionAsOf", 0).load(delta_table_path)
   new_df.show(10)
   ```

1. The log contains a full history of modifications to the data. Use the following code to see a record of the last 10 changes:

   ```python
   deltaTable.history(10).show(10, False, True)
   ```

## Create catalog tables

So far you've worked with delta tables by loading data from the folder containing the parquet files on which the table is based. You can define _catalog tables_ that encapsulate the data and provide a named table entity that you can reference in SQL code. Spark supports two kinds of catalog tables for delta lake:

- _External_ tables that are defined by the path to the files containing the table data.
- _Managed_ tables, that are defined in the metastore.

### Create a managed table

1. Run the following code to create (and then describe) a managed table named **ProductsManaged** based on the dataframe you originally loaded from the **products.csv** file (before you updated the price of product 771).

   ```python
   df.write.format("delta").saveAsTable("AdventureWorks.ProductsManaged")
   spark.sql("DESCRIBE EXTENDED AdventureWorks.ProductsManaged").show(truncate=False)
   ```

   You did not specify a path for the parquet files used by the table - this is managed for you in the Hive metastore, and shown in the **Location** property in the table description.

1. Use the following code to query the managed table, noting that the syntax is just the same as for a managed table:

   ```sql
   %sql
   USE AdventureWorks;
   SELECT * FROM ProductsManaged;
   ```

1. Use the following code to delete table from the database:

   ```sql
   %sql
   USE AdventureWorks;
   DROP TABLE IF EXISTS ProductsManaged;
   SHOW TABLES;
   ```
