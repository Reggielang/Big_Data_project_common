
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix

/**
 * Product数据集
 * 3982                            商品ID
 * Fuhlen 富勒 M8眩光舞者时尚节能    商品名称
 * 1057,439,736                    商品分类ID，不需要
 * B009EJN4T2                      亚马逊ID，不需要
 * https://images-cn-4.ssl-image   商品的图片URL
 * 外设产品|鼠标|电脑/办公           商品分类
 * 富勒|鼠标|电子产品|好用|外观漂亮   商品UGC标签
 */
case class Product( productId: Int, name: String, imageUrl: String, categories: String, tags: String )


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


object ContentRecommender {
  //定义一些常量
  // 定义mongodb中存储的表名
  val MONGODB_PRODUCT_COLLECTION = "Product"
  //定义推荐列表的表名称
  val CONTENT_PRODUCT_RECS="ContentBasedProductRecs"


  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://hadoop101:27017/recommender",
      "mongo.db" -> "recommender"
    )
    // 创建一个spark config
    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("ContentRecommender")
    // 创建spark session
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    //RDD和DF,DS相互转换的必要导入
    import spark.implicits._
    implicit val mongoConfig = MongoConfig( config("mongo.uri"), config("mongo.db") )

    //加载数据
    val productTagsDF = spark.read
      .option("uri",mongoConfig.uri)
      .option("collection",MONGODB_PRODUCT_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Product]
      .map(
        //tags分隔符，替换为空格
        x=>(x.productId,x.name,x.tags.map(c=> if(c=='|') ' ' else c))
      )
      .toDF("productId","name","tags")
      .cache()


    //todo TF-IDF算法提取商品特征向量
    // 1.实例化一个分词器，用来实现分词,默认按照空格分词
    val tokenizer = new Tokenizer().setInputCol("tags").setOutputCol("words")

    // 2.用分词器做转换，得到增加一个新列words的DF
    val wordsDataDF = tokenizer.transform(productTagsDF)

    //3.定义一个HashingTF工具，计算频次
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rowFeatures")

    val featurizedDataDF = hashingTF.transform(wordsDataDF)

    // 4.定义IDF，计算TF-IDF
    val idf = new IDF().setInputCol("rowFeatures").setOutputCol("features")

    //训练一个idf模型
    val idfModel = idf.fit(featurizedDataDF)

    //得到增加新列feature的DF
    val rescaledDataDF = idfModel.transform(featurizedDataDF)


    //对数据进行转换，得到RDD形式的features
    val productFeatures =rescaledDataDF.map{
      row=>(row.getAs[Int]("productId"),row.getAs[SparseVector]("features").toArray)
    }
      .rdd
      .map{
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
      .option("collection",CONTENT_PRODUCT_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    spark.stop()
    println("ContentRecommender over !!!")
  }

  //计算余弦相似度函数
  def consinSim(product1: DoubleMatrix, product2: DoubleMatrix):Double = {
    product1.dot(product2)/(product1.norm2() * product2.norm2())
  }
}
