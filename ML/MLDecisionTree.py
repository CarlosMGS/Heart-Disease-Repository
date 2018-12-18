
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("MachineLearning").getOrCreate()

df = spark.read.option("header","false").option("delimiter"," ").csv("heart.txt")


data= df.selectExpr("cast(_c0 as double) as age","cast(_c1 as double) as sex","cast(_c2 as double) as cp","cast(_c3 as double) as trestbps","cast(_c4 as double) as chol","cast(_c5 as double) as fbs","cast(_c6 as double) as restecg","cast(_c7 as double) as thalach","cast(_c8 as double) as exang","cast(_c9 as double) as oldpeak","cast(_c10 as double) as slope","cast(_c11 as double) as ca","cast(_c12 as double) as thal","cast(_c13 as int) as num")
cols=data.columns
stages=[]

label_stringIdx = StringIndexer(inputCol="num", outputCol="label")
stages += [label_stringIdx]

numeric_cols=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]

assemblerInputs = numeric_cols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(data)
data = pipelineModel.transform(data)
selectedCols = ['label','features'] + cols
data = data.select(selectedCols)
data.printSchema()

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

# Chain indexer and tree in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, dt])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

treeModel = model.stages[1]
# summary only
print(treeModel)
