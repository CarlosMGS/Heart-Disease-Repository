from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# carga los datos de entrenamiento
data = spark.read.format("libsvm") \
    .load("dataset.txt")

# separa los datos en entrenaiento y test
(train, test) = data.randomSplit([0.6, 0.4])

# crea el entrenador y sus parametros
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# entrena el modelo
model = nb.fit(train)

# comprueba con el modelo con los test
predictions = model.transform(test)
predictions.show()

# computa la precisión del modelo
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Precisión del conjunto del test = " + str(accuracy))