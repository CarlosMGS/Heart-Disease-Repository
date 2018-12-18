from pyspark.ml.classification import LinearSVC

# carga los datos
data = spark.read.format("libsvm").load("dataset.txt")
(train, test) = data.randomSplit([0.7, 0.3])
lsvc = LinearSVC(maxIter=10, regParam=0.1)

# ejecuta el entrenamiento
lsvcModel = lsvc.fit(train)

# ejecuta el test
predictions = ovrModel.transform(test)

# obtén evaluador
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

# computa el porcentage de error
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

# imprime el resultado 
print("Coefficients: " + str(lsvcModel.coefficients))
print("Intercept: " + str(lsvcModel.intercept))