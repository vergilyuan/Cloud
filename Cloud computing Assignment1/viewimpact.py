import findspark
findspark.init()
from pyspark import SparkContext
from utils import *
import argparse
import csv

if __name__ == "__main__":
    sc = SparkContext(appName="Video trending impact")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="the input path",
                        default='hdfs://localhost:9000/user/shengyuan/')
    parser.add_argument("--output", help="the output path", 
                        default='hdfs://localhost:9000/user/shengyuan/output2') 
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    inputData = sc.textFile(input_path + "ALLvideos.csv")
    VideoData = inputData.mapPartitions(extractVideo)
    GroupData=VideoData.groupByKey().mapValues(list) 
    ViewsData = GroupData.flatMap(extractViews)
    DescendingData=ViewsData.sortBy(ascending=False, keyfunc=lambda k: k[1]).groupBy(SortCountry).mapValues(list)
    OutputData=DescendingData.flatMap(SortMap)
    OutputData.saveAsTextFile(output_path)

