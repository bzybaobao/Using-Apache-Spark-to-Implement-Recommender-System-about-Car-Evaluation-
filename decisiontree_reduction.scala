import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.Row
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

val data = spark.read.format("libsvm").load("/FileStore/tables/bx72m8tz1490406519554/rst.data")
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(3).fit(data)
val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2))
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
val model = pipeline.fit(trainingData)
val predictions = model.transform(testData)
predictions.select("predictedLabel", "label", "features").show(5)
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println("Learned classification tree model:\n" + treeModel.toDebugString)
val predictions1 = model.transform(testData)
val predictionAndLabelsRDD = predictions1.select("predictedLabel", "label").rdd;
val predictionAndLabels = predictionAndLabelsRDD.map{
   case Row(key: String, value: Double) =>
     key.toDouble -> value
 }
val metrics = new MulticlassMetrics(predictionAndLabels)
val metrics1 = new BinaryClassificationMetrics(predictionAndLabels)
val auROC = metrics1.areaUnderROC
println("Area under ROC = " + auROC)
val auPRC = metrics1.areaUnderPR
println("Area under precision-recall curve = " + auPRC)
val accuracy1 = metrics.accuracy
println(s"Accuracy1 = $accuracy1")
val trainerror=1-accuracy1
println(s"trainerror = $trainerror")

