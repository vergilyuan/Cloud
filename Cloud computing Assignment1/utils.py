import csv
def extractVideo(record):
    try:
        line = csv.reader(record,delimiter=",")

        for x in line: 
            Vid=x[0]
            Views=x[8]
            Country=x[17]
            Combine=Country+";"+Vid
            if Vid!="video_id":
                yield (Combine,Views)
    except:
        return ()

def extractViews(record):
    try:
        name, ViewsList = record
        view=list (ViewsList)
        if len(view)>1:
            trending=float ((int (view[1])-int (view[0]))/int (view[0]))
            trending=round(trending*100,2)
            if trending>1000:
                output=str(trending)+"%"
                yield name,output
    except: 
        return

def SortCountry(record):
    try:
        name, ViewsList = record
        country=name[0]+name[1]
        return country
    except:
        return 

def SortMap(record):
    try:
        country, outputList =record
        return outputList
    except:
        return