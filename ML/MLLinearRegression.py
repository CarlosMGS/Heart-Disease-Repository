
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
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

lr = LinearRegression(maxIter=1000, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(data)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

