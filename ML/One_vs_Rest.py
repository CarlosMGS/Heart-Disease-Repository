from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# carga el fichero
inputData = spark.read.format("libsvm") \
    .load("dataset.txt")

# genera la división de entrenamiento y test
(train, test) = inputData.randomSplit([0.8, 0.2])

# instancia la base clasificador
lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)

# instancia el algoritmo OnevsRest
ovr = OneVsRest(classifier=lr)

# entrena el modelo multiclase
ovrModel = ovr.fit(train)

# ejecuta el test
predictions = ovrModel.transform(test)

# obtén evaluador
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

# computa el porcentage de error
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))