# Importing Necessary Libraries
import os
from os import listdir
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import osmnx as ox
from descartes import PolygonPatch
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from mumbaiVirus import *
from pyproj import CRS
import contextily as ctx
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm_notebook

def seirPlot(result): #F
        # Plotting Every 12th value per iteration to account for 1 day. (Each time step is 2 hrs).
        plt.plot(result[::12, 0], color='r', label='S')
        plt.plot(result[::12, 1], color='g', label='E')
        plt.plot(result[::12, 2], color='b', label='I')
        plt.plot(result[::12, 3], color='y', label='R')
        plt.plot(result[::12, 4], color='c', label='H')
        plt.legend()
        plt.show()

with open(os.path.abspath("covData/Yerevan_population.pkl"), "rb") as population:
    mumbaiPopulation = pickle.load(population)[:,:531]

with open(os.path.abspath("covData/Yerevan_OD_matrices.pkl"), "rb") as originDestinationMatrix:
        originDestinationMatrix = pickle.load(originDestinationMatrix)[:,:531,:531]

timeStep = originDestinationMatrix.shape[0] #84 time steps
cells = mumbaiPopulation.shape[1] #531 cells
totalPopulation = 1000000.0 #The population of Mumbai is approx 12.5 million, but for the purpose of this simulation, I've set the population as 1 million

initialInd = [334,353,196,445,162,297]
initial = np.zeros(cells)
initial[initialInd] = 100

model = par(r0Number = 2.4, DE = 5.6 * 12, infectiousPeriod = 5.2 * 12, initialRandomInfections = initial,  HospitalisationRate = 0.1, timeInHospital = 15 * 12)

# lockdown strictness = 1, ie no lockdowns
lockdownStrictness = np.ones(originDestinationMatrix.shape)
iterations = 3000
result = {}
initialInfections = 100
result['baseline'] = seirModel(model, mumbaiPopulation, originDestinationMatrix, lockdownStrictness, iterations, initialInfections)

print("Max Number of hospitalised people: ", int(result["baseline"][0][:, 4].max()),
       "\n",
       "Day with max hospitalised people: ", int(result['baseline'][0][:, 4].argmax()/12))
seirPlot(result["baseline"][0])

# Now Creating the Spacial Visualisation
# Setting the CRS to 4326
crs = CRS.from_epsg(4326)

# As referenced in the geopandas into file, we need to first laod the Mumbai City

mumbaiCity = ox.geocode_to_gdf('Mumbai, India')
mumbaiCity = ox.projection.project_gdf(mumbaiCity)
ax = mumbaiCity.plot(ec='none')
_ = ax.axis('off')
geometry = mumbaiCity['geometry'].iloc[0]
geometry_cut = ox.utils_geo._quadrat_cut_geometry(geometry, quadrat_width = 1050)
polygonList = [p for p in geometry_cut]
west, south, east, north = mumbaiCity.unary_union.bounds
fig, ax = plt.subplots(figsize=(20,20), dpi=50)

for polygon, n in zip(geometry_cut, np.arange(len(polygonList))):
    polygonCentre = polygon.representative_point().coords[:][0]
    patch = PolygonPatch(polygon, fc='#ffffff', ec='#000000', alpha=0.5, zorder=2)
    ax.add_patch(patch)
    plt.annotate(xy=polygonCentre, text=n, horizontalalignment='center', size=12)

ax.set_xlim(west, east)
ax.set_ylim(south, north)
ax.axis('off')

polygonFrame = gpd.GeoDataFrame(geometry=polygonList)
polygonFrame.crs = mumbaiCity.crs
polygonFrame = polygonFrame.to_crs(epsg=3857)
west, south, east, north = polygonFrame.unary_union.bounds
ax = polygonFrame.plot(figsize=(10,10), alpha=0.5, edgecolor='k')
ctx.add_basemap(ax, zoom='auto')
ax.set_xlim(west, east)
ax.set_ylim(south, north)

polygonFrame = polygonFrame.to_crs(epsg=3857)
west, south, east, north = polygonFrame.unary_union.bounds
baseline = result['baseline'][1][::12, :, :]

hospitalisation = result['baseline'][0][::12, 4]

# find maximum hospitalisation value to make sure the color intensities in the animation are anchored against it
max_exp_ind = np.where(baseline[:, 1, :] == baseline[:, 1, :].max())[0].item()
max_exp_val = baseline[:, 1, :].max()

print(max_exp_ind, max_exp_val)

ncolors = 256
color_array = plt.get_cmap('Reds')(range(ncolors))

plt.rcParams.update({"axes.labelcolor":"slategrey"})
cmap = plt.cm.get_cmap("Blues")
blue = cmap(200)

for time_step in tqdm_notebook(range(1,251)):
    polygonFrame['exposed'] = baseline[time_step-1, 1, :]
    fig, ax = plt.subplots(figsize=(14, 14), dpi=72)
    # Picked 32, because it is so small, it is unrecognisable
    polygonFrame.loc[polygonFrame.index == 32, 'exposed'] = max_exp_val + 1
    polygonFrame.plot(ax=ax, facecolor='none', edgecolor='gray', alpha=0.5, linewidth=0.5, zorder=2)
    polygonFrame.plot(ax=ax, column='exposed', cmap='Reds', zorder=3)

    ctx.add_basemap(ax, attribution="", url=ctx.sources.ST_TONER_LITE, zoom='auto', alpha=0.6)

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.axis('off')
    plt.tight_layout()

    inset_ax = fig.add_axes([0.6, 0.14, 0.37, 0.27])
    inset_ax.patch.set_alpha(0.5)

    inset_ax.plot(baseline[:time_step, 0].sum(axis=1), label="susceptible", color=blue, ls='-', lw=1.5, alpha=0.8)
    inset_ax.plot(baseline[:time_step, 1].sum(axis=1), label="exposed", color='g', ls='-', lw=1.5, alpha=0.8)
    inset_ax.plot(baseline[:time_step, 2].sum(axis=1), label="infectious", color='r', ls='-', lw=1.5, alpha=0.8)
    inset_ax.plot(baseline[:time_step, 3].sum(axis=1), label="recovered", color='y', ls='-', lw=1.5, alpha=0.8)
    inset_ax.plot(hospitalisation[:time_step], label="hospitalised", color='purple', ls='-', lw=1.5, alpha=0.8)

    inset_ax.scatter((time_step-1), baseline[(time_step-1), 0].sum(), color=blue, s=50, alpha=0.2)
    inset_ax.scatter((time_step-1), baseline[(time_step-1), 1].sum(), color='g', s=50, alpha=0.2)
    inset_ax.scatter((time_step-1), baseline[(time_step-1), 2].sum(), color='r', s=50, alpha=0.2)
    inset_ax.scatter((time_step-1), baseline[(time_step-1), 3].sum(), color='y', s=50, alpha=0.2)
    inset_ax.scatter((time_step-1), hospitalisation[(time_step-1)], color='purple', s=50, alpha=0.2)

    inset_ax.fill_between(np.arange(0, time_step), np.maximum(baseline[:time_step, 0].sum(axis=1), \
                                                              baseline[:time_step, 3].sum(axis=1)), alpha=0.035,
                          color='r')
    inset_ax.plot([time_step, time_step], [0, max(baseline[(time_step - 1), 0].sum(), \
                                                  baseline[(time_step - 1), 3].sum())], ls='--', lw=0.7, alpha=0.8,
                  color='r')

    inset_ax.set_ylabel('Population', size=18, alpha=1, rotation=90)
    inset_ax.set_xlabel('Days', size=18, alpha=1)
    inset_ax.yaxis.set_label_coords(-0.15, 0.55)
    inset_ax.tick_params(direction='in', size=10)
    inset_ax.set_xlim(-4, 254)
    inset_ax.set_ylim(-24000, 1024000)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    inset_ax.grid(alpha=0.4)

    inset_ax.spines['right'].set_visible(False)
    inset_ax.spines['top'].set_visible(False)

    inset_ax.spines['left'].set_color('darkslategrey')
    inset_ax.spines['bottom'].set_color('darkslategrey')
    inset_ax.tick_params(axis='x', colors='darkslategrey')
    inset_ax.tick_params(axis='y', colors='darkslategrey')
    plt.legend(prop={'size': 14, 'weight': 'light'}, framealpha=0.5)

    plt.title("Mumbai Covid-19 spreading on day: {}".format(time_step), fontsize=18, color='dimgray')

    # plt.savefig("MumbaiPlots/flows_{}.jpg".format(time_step), dpi=72)

# ----------------------------

def sort_in_order(l): #F
    """sorts a given iterable where
    l is the iterable to be sorted"""

    convert = lambda text: int(text) if text.isdigit() else text
    alphanumeric_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanumeric_key)


filenames = listdir("MumbaiPlots/")
filenames = sort_in_order(filenames)
print(filenames)

import imageio

with imageio.get_writer('Mumbai.gif', mode='I', fps=16) as writer:
    for filename in tqdm_notebook(filenames):
        image=imageio.imread('MumbaiPlots/{}'.format(filename))
        writer.append_data(image)