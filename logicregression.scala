var data =sc.textFile("/FileStore/tables/xxs099ho1490401002441/car.data")
data.count()
data.collect().foreach(println)
def getDoubleValue( input:String ) : Double = {
    var result:Double = 0.0
    if (input == "vhigh")  result = 4.0 
    if (input == "high")  result = 3.0
    if (input == "med")  result = 2.0
    if (input == "low")  result = 1.0
    if (input == "5-more")  result = 5.0
    if (input == "more")  result = 5.0
    if (input == "big")  result = 3.0 
    if (input == "small")  result = 1.0 
    if (input == "unacc") result = 1.0
    if (input == "acc")  result = 0.0
    return result
   }

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

val parsedData = data.map{line => 
    val parts = line.split(",")
    LabeledPoint(getDoubleValue(parts(6)), Vectors.dense(parts.slice(0,6).map(x => getDoubleValue(x))))
}

parsedData.collect().foreach(println)

val splits = parsedData.randomSplit(Array(0.8, 0.2), seed = 11L)

splits(0).collect.foreach(println)

val trainingData = splits(0)
val testData = splits(1)

val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainingData)

val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
abelAndPreds.collect.foreach(println)

val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count

println("prediction = " +  (1-trainErr))

val metrics = new BinaryClassificationMetrics(labelAndPreds)

val f1Score = metrics.fMeasureByThreshold
f1Score.foreach { case (t, f) =>
  println(s"Threshold: $t, F-score: $f, Beta = 1")
}

val auPRC = metrics.areaUnderPR
println("Area under precision-recall curve = " + auPRC)

val auROC = metrics.areaUnderROC
println("Area under ROC = " + auROC)

import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

val pca = new PCA(2).fit(trainingData.map(_.features))

val train_pca = trainingData.map(p => p.copy(features = pca.transform(p.features)))

val test_pca = testData.map(p => p.copy(features = pca.transform(p.features)))

val model1 = new LogisticRegressionWithLBFGS().setNumClasses(2).run(train_pca)

val labelAndPreds1 = test_pca.map { point =>
  val prediction1 = model1.predict(point.features)
  (point.label, prediction1)
}

val trainErr1 = labelAndPreds1.filter(r => r._1 != r._2).count.toDouble / test_pca.count

println("prediction = " +  (1-trainErr1))

val metrics1 = new BinaryClassificationMetrics(labelAndPreds1)

val auPRC1 = metrics1.areaUnderPR
println("Area under precision-recall curve = " + auPRC1)

val auROC1 = metrics1.areaUnderROC
println("Area under ROC = " + auROC1)