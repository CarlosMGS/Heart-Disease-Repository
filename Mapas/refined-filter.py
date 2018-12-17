#!/usr/bin/python

from pyspark import SparkConf, SparkContext
import string

import sys

pattern = sys.argv[1]
s_name = sys.argv[1].replace(" ", "")
name = s_name + '.csv'

def tocsv(line):
    return ','.join(str(word) for word in line)

conf = SparkConf().setMaster('local').setAppName('app.py')
sc = SparkContext(conf = conf)

RDDvar = sc.textFile("risks-f.csv")

RDDvar = RDDvar.map(lambda line: line.split(","))

RDDvar = RDDvar.filter(lambda cols: cols[1] == pattern)
RDDvar = RDDvar.map(lambda cols: (cols[0], float(cols[2])))

RDDvar = RDDvar.groupByKey()

RDDvar = RDDvar.map(lambda elem: (elem[0], float(sum(list(elem[1]))) / len(elem[1])))

RDDvar = RDDvar.map(tocsv)
RDDvar.saveAsTextFile(name)