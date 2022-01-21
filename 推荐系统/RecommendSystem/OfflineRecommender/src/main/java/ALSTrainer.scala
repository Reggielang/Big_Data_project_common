import OfflineRecommender.MONGODB_RATING_COLLECTION
import breeze.numerics.sqrt
import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object ALSTrainer {
  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://hadoop101:27017/recommender",
      "mongo.db" -> "recommender"
    )
    // 创建一个spark config
    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("OfflineRecommender")
    // 创建spark session
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    //RDD和DF,DS相互转换的必要导入
    import spark.implicits._
    implicit val mongoConfig = MongoConfig( config("mongo.uri"), config("mongo.db") )

    //加载数据
    val ratingRDD=spark.read
      .option("uri",mongoConfig.uri)
      .option("collection",MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[ProductRating]
      .rdd
      .map(
        rating=>{
          Rating(rating.userId,rating.productId,rating.score)
        }
      ).cache()
    //数据集分成训练集，测试集
    val splits = ratingRDD.randomSplit(Array(0.7,0.3))
    val trainRDD = splits(0)
    val testRDD = splits(1)

    //核心实现，输出最优参数
    adjustALSParams(trainRDD,testRDD)

    spark.stop()
  }

  def adjustALSParams(trainData: RDD[Rating], testData: RDD[Rating])={
    //用循环遍历不同的参数组合
    val result = for(rank<-Array(5,10,20,30,40,50);lambda<-Array(1,0.1,0.01))
      yield{
        val model = ALS.train(trainData,rank,10,lambda)
        val rmse = getRMSE(model,testData)
        (rank,lambda,rmse)
      }
    //按照rmse排序，输出最优参数
    println(result.minBy(_._3))
  }

  def getRMSE(model: MatrixFactorizationModel, data: RDD[Rating]):Double={
    //构建userProducts,得到预测评分矩阵
    val userProducts = data.map(item=>(item.user,item.product))
    val predictRating = model.predict(userProducts)

    //按照公式计算rmse 把真实值，和预测值以userid和productid做连接
    val observed = data.map(item=>((item.user,item.product),item.rating))
    val predict = predictRating.map(item=>((item.user,item.product),item.rating))


    sqrt(observed.join(predict).map{
      case ((userId, productId),(actual,pre)) =>
        val err = actual - pre
        err*err
    }.mean())
  }
}
