import pandas as pd
from pathlib import Path
from scipy.io import arff
import pkg_resources
from io import StringIO

def getCreditData():
    s_io=StringIO(str(pkg_resources.resource_string('pysubgroup', 'data/credit-g.arff'),'utf-8'))
    return  pd.DataFrame(arff.loadarff(s_io)[0])

def getTitanicData():
    s_io=StringIO(str(pkg_resources.resource_string('pysubgroup', 'data/titanic.csv'),'utf-8'))
    return pd.read_csv(s_io,sep="\t",header=[0])