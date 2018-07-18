#!/bin/bash

if [ $# -ne 4 ]; then
    echo "Invalid number of parameters!"
    echo "Usage: ./tag_driver.sh [input_location] [output_location]"
    exit 1
fi

hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-3.0.0.jar \
-D mapreduce.job.reduces=3 \
-D mapreduce.job.name='Assignment1_3.1' \
-file video_mapper.py \
-mapper "python video_mapper.py $1 $2" \
-file video_reducer.py \
-reducer "python video_reducer.py $1 $2" \
-input $3 \
-output $4