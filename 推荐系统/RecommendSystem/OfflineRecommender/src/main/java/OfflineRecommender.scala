import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix

/*
基于隐语义模型的协同过滤推荐
项目采用ALS作为协同过滤算法，根据MongoDB中的用户评分表计算离线的用户商品推荐列表以及商品相似度矩阵。

用户商品推荐列表
通过ALS训练出来的Model来计算所有当前用户商品的推荐列表，主要思路如下：
1.userId和productId做笛卡尔积，产生（userId，productId）的元组
2.通过模型预测（userId，productId）对应的评分。
3.将预测结果通过预测分值进行排序。
4.返回分值最大的K个商品，作为当前用户的推荐列表。
 */

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
//定义用户的推荐列表
case class UserRecs(userId:Int,recs:Seq[Recommendation])
//定义商品相似度列表
case class ProductRecs(ProductId:Int,recs:Seq[Recommendation])


object OfflineRecommender {
  // 定义mongodb中存储的表名
  val MONGODB_RATING_COLLECTION = "Rating"

  //定义用户推荐列表的表名称
  val USER_RECS="UserRecs"
  val PRODUCT_RECS="ProductRecs"
  //最大推荐个数
  val USER_MAX_RECOMMENDATION=20



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
          (rating.userId,rating.productId,rating.score)
        }
      ).cache()

    //提取出所有用户和商品的数据集
    val userRDD = ratingRDD.map(_._1).distinct()
    val ProductRDD = ratingRDD.map(_._2).distinct()

    //todo:计算过程
    // 1.训练隐语义模型ALS 数据转换为mllib里的Rating结构
    val trainData = ratingRDD.map(productRating=>Rating(productRating._1,productRating._2,productRating._3))
    // 定义模型训练的参数 rank=隐特征个数，iterations=迭代次数 lambda=正则化系数
    val (rank,iterations,lambda) = (5,10,0.01)
    val model = ALS.train(trainData,rank,iterations,lambda)

    // 2.获得预测评分矩阵，得到用户的推荐列表
    //用userRDD 和productRDD做笛卡尔积，得到空的userProductsRDD
    val userProducts= userRDD.cartesian(ProductRDD)
    val preRating = model.predict(userProducts)

    //从预测评分矩阵中，得到用户推荐列表
    val userRecs = preRating.filter(_.rating>0)
      .map(
        rating=>(rating.user,(rating.product,rating.rating))
      )
      .groupByKey()
      .map{
        case (userId,recs)=>
          UserRecs(userId,recs.toList.sortBy(_._2)(Ordering.Double.reverse).take(USER_MAX_RECOMMENDATION).map(x=>Recommendation(x._1,x._2)))
      }
      .toDF()

    userRecs.write
      .option("uri",mongoConfig.uri)
      .option("collection",USER_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    // 3.利用商品的特征向量，计算商品的相似度列表
    val productFeatures = model.productFeatures.map{
      case (productId, features) =>(productId,new DoubleMatrix(features))
    }

    //两两配对商品，计算余弦相似度
    //自己跟自己做笛卡尔积，剔除掉自己
    val productRecs = productFeatures.cartesian(productFeatures)
      .filter{
      case(a,b)=>a._1!= b._1
      }
      //计算余弦相似度
      .map{
        case(a,b)=>
          val simScore = consinSim(a._2,b._2)
          (a._1,(b._1,simScore))
      }
      .filter(_._2._2>0.4)
      .groupByKey()
      .map{
        case (productId,recs)=>
          ProductRecs(productId,recs.toList.sortBy(_._2).map(x=>Recommendation(x._1,x._2)))
      }
      .toDF()
    productRecs.write
      .option("uri",mongoConfig.uri)
      .option("collection",PRODUCT_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    spark.stop()
  }

  //计算余弦相似度函数
  def consinSim(product1: DoubleMatrix, product2: DoubleMatrix):Double = {
    product1.dot(product2)/(product1.norm2() * product2.norm2())
  }
}
