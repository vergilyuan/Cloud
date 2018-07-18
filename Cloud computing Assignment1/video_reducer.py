#!/usr/bin/python3

import sys

def read_map_output(file):
    for line in file:
        yield line.strip().split("\t")


def video_reducer():
    current_cate=""
    current_name=""
    output_A=list()
    output_B=list()
    A=sys.argv[1]
    B=sys.argv[2]
    for category,cate_name, country,value in read_map_output(sys.stdin):
        if category == current_cate:
            if(country == A):
                output_A.append(value)
            elif(country == B):
                output_B.append(value)
        else:
            if current_cate:
                re_A=list(set(output_A))
                re_B=list(set(output_B))
                AB=list(set(re_A).intersection(set(re_B)))
                if len(re_A)>0:
                    ratio=round(float (len(AB))/float (len(re_A)),4)
                    print("{};\ttotal:{};\t{}%\tin\t{}".format(current_name,len(re_A),ratio*100,B))
                else:
                    ratio=0
                    print("{};\ttotal:{};\t{}%\tin\t{}".format(current_name,len(re_A),ratio*100,B))
            current_cate=category
            current_name=cate_name
            output_A=list()
            output_B=list()
    if current_cate:
        re_A=list(set(output_A))
        re_B=list(set(output_B))
        AB=list(set(re_A).intersection(set(re_B)))
        if len(re_A)>0:
            ratio=round(float (len(AB))/float (len(re_A)),4)
            print("{};\ttotal:{};\t{}%\tin\t{}".format(current_name,len(re_A),ratio*100,B))
        else:
            ratio=0
            print("{};\ttotal:{};\t{}%\tin\t{}".format(current_name,len(re_A),ratio*100,B))


if __name__ == "__main__":
    video_reducer()