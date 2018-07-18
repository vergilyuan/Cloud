#!/usr/bin/python3
import sys , csv 
def video_mapper():
    A=sys.argv[1]
    B=sys.argv[2]
    cfile = csv.reader(sys.stdin,delimiter=",")
    for line in cfile:
        vid=line[0]
        category=line[4]
        cate_name=line[5]
        country=line[17]
        if(country == A or country == B):
            print("{}\t{}\t{}\t{}".format(category, cate_name, country,vid))


if __name__== "__main__":
    video_mapper()