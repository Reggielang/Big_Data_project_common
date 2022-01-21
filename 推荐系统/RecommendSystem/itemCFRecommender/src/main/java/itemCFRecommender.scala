import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

/**
 * Rating数据集
 * 4867        用户ID
 * 457976      商品ID
 * 5.0         评分
 * 1395676800  时间戳
 */
case class ProductRating( userId: Int, productId: Int, score: Double, timestamp: Int )

/**
 * MongoDB连接配置
 * @param uri    MongoDB的连接uri
 * @param db     要操作的db
 */
case class MongoConfig( uri: String, db: String )

//定义标准的推荐对象
case class Recommendation(productId:Int,score:Double)

//定义商品相似度列表
case class ProductRecs(ProductId:Int,recs:Seq[Recommendation])

object itemCFRecommender {
    //定义常量和表名
    val MONGODB_RATING_COLLECTION = "Rating"

    //定义用户推荐列表的表名称
    val ITEM_CF_PRODUCT_RECS="ItemCFProductRecs"
    //最大推荐个数
    val MAX_RECOMMENDATION=10

  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://hadoop101:27017/recommender",
      "mongo.db" -> "recommender"
    )
    // 创建一个spark config
    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("ItemCFRecommender")
    // 创建spark session
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    //RDD和DF,DS相互转换的必要导入
    import spark.implicits._
    implicit val mongoConfig = MongoConfig( config("mongo.uri"), config("mongo.db") )

    //加载数据，转换成DF进行处理
    val ratingDF = spark.read
      .option("uri",mongoConfig.uri)
      .option("collection",MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[ProductRating]
      .map(
        x=>(x.userId,x.productId,x.score)
      )
      .toDF("userId","productId","score")
      .cache()

    //todo 核心算法 计算同现相似度，得到商品的相似度列表
    // 1.统计每个商品的评分个数，按照productId做group by
    val productRatingCountDF = ratingDF.groupBy("productId").count()

    // 2.在原有的评分表rating中添加count
    val ratingWithCountDF = ratingDF.join(productRatingCountDF,"productId")

    // 3.将评分按照用户Id两两配对，统计两个商品被同一个用户评分过的个数
    val joinedDF = ratingWithCountDF.join(ratingWithCountDF,"userId")
      .toDF("userId","product1","score1","count1","product2","score2","count2")
      .select("userId","product1","count1","product2","count2")
    // 4.创建一张临时表，用于写SQL查询
    joinedDF.createOrReplaceTempView("joined")

    // 5.按照productId1，productId2,做group by统计userId的数量，就是对两个商品同时评分的人数
    val coocurrenceDF = spark.sql(
      """
        |select product1,product2,count(userId) as cocount,
        |first(count1) as count1, first(count2) as count2 from joined
        |group by product1,product2
        |""".stripMargin
    ).cache()


    //6. 提取需要的数据，包装成（productId1,(productId2,score)）
    val simDF = coocurrenceDF.map{
      row =>
        val coocSim = coocurrenceSim(row.getAs[Long]("cocount"),row.getAs[Long]("count1"),row.getAs[Long]("count2"))
        (row.getInt(0),(row.getInt(1),coocSim))
    }
      .rdd
      .groupByKey()
      .map{
        case (productId,recs)=>
          ProductRecs(productId,recs.toList
            .filter(x=>x._1 != productId)
            .sortBy(_._2)
            .take(MAX_RECOMMENDATION)
            .map(x=>Recommendation(x._1,x._2)))
      }
      .toDF()

    //保存到mongoDB
    simDF.write
      .option("uri",mongoConfig.uri)
      .option("collection",ITEM_CF_PRODUCT_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    spark.stop()
    println("ItemCF over!!!")
  }

  //按照公式计算同现相似度
  def coocurrenceSim(coCount:Long,count1:Long,count2:Long):Double={
    coCount/math.sqrt(count1*count2)
  }
}
