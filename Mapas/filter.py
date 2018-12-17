#!/usr/bin/python

from pyspark import SparkConf, SparkContext
import string



primera = True

def primeralinea():
    global primera
    
    if primera:
        primera = False
        return True
    else:
		return False

def tocsv(line):
    return ','.join(str(word) for word in line)

conf = SparkConf().setMaster('local').setAppName('P24_spark.py')
sc = SparkContext(conf = conf)

RDDvar = sc.textFile("risks.csv")

RDDvar = RDDvar.filter(lambda line: not primeralinea())

RDDvar = RDDvar.map(lambda line: line.split(","))
RDDvar = RDDvar.map(lambda cols: (cols[1], cols[9], cols[15]))


RDDvar = RDDvar.map(tocsv)
RDDvar.saveAsTextFile('risks-filtered.csv')






#first = RDDvar.take(5)

#print first


