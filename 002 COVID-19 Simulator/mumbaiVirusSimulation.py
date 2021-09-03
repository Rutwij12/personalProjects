import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import osmnx as ox
from descartes import PolygonPatch
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from mumbaiVirusSim import *

def seir_plot(res):
    plt.plot(res[::12, 0], color = 'r', label = 'S')
    plt.plot(res[::12, 1], color = 'g', label = 'E')
    plt.plot(res[::12, 2], color = 'b', label = 'I')
    plt.plot(res[::12, 3], color = 'y', label = 'R')
    plt.plot(res[::12, 4], color = 'c', label = 'H')
    plt.legend()

pkl_file = open("covData/Yerevan_OD_matrices.pkl", "rb")
OD_matrices = pickle.load(pkl_file)[:,:531,:531]
print(OD_matrices)
OD_matrices.shape

np.set_printoptions(suppress=True, precision=3)

pkl_file = open("covData/Yerevan_population.pkl", "rb")
pop = pickle.load(pkl_file)[:,:531]
pkl_file.close()
print(pop)
print(len(pop))


r = OD_matrices.shape[0]
n = pop.shape[1]
N = 1000000.0

initialInd = [334,353,196,445,162,297]
initial = np.zeros(n)
initial[initialInd] = 50

model = Param(R0 =2.4, DE = 5.6 * 12, DI = 5.2 * 12, I0 = initial, HospitalisationRate = 0.1, HospitalIters = 15 * 12)

alpha = np.ones(OD_matrices.shape)
iterations = 3000
res = {}
inf = 50
res['baseline'] = seir(model, pop, OD_matrices, alpha, iterations, inf)

print(
"Max number of hospitalised people: ", int(res["baseline"][0][:,4].max()),
"\n",
"Day with max hospitalised people: ", int(res["baseline"][0][:,4].argmax()/12)
)

seir_plot(res["baseline"][0])