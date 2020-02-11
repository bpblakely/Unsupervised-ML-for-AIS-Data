# -*- coding: utf-8 -*-
"""
Utility functions for working with AIS data. Also, custom functions built in for algorithm and data analysis.

@author: Brian Blakely
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers,colors
from mpl_toolkits.mplot3d import Axes3D
from gmplot import gmplot

def convertTimeToSec(timeVec):
    # Convert time from hh:mm:ss string to number of seconds
    return sum([a * b for a, b in zip(
            map(int, timeVec.decode('utf-8').split(':')), [3600, 60, 1])])

def loadData(filename):
    # Load data from CSV file into numPy array, converting times to seconds
    timestampInd = 2

    data = np.loadtxt(filename, delimiter=",", dtype=float, skiprows=1, 
                      converters={timestampInd: convertTimeToSec})

    return data

def plotVesselTracks(latLon, clu=None):
    # Plot vessel tracks using different colors and markers with vessels
    # given by clu
    
    n = latLon.shape[0]
    if clu is None:
        clu = np.ones(n)
    cluUnique = np.array(np.unique(clu), dtype=int)
    
    plt.figure()
    markerList = list(markers.MarkerStyle.markers.keys())
    
    normClu = colors.Normalize(np.min(cluUnique),np.max(cluUnique))
    for iClu in cluUnique:
        objLabel = np.where(clu == iClu)
        imClu = plt.scatter(
                latLon[objLabel,0].ravel(), latLon[objLabel,1].ravel(),
                marker=markerList[iClu % len(markerList)],
                c=clu[objLabel], norm=normClu, label=iClu)
    plt.colorbar(imClu)
    plt.legend().set_draggable(True)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
# prints given coordinates on google maps
def gmapVessel(latLon,writeName,clu=None):
    n = latLon.shape[0]
    if clu is None:
        clu = np.ones(n)
    cluUnique = np.array(np.unique(clu), dtype=int)
    gmap= gmplot.GoogleMapPlotter(np.sum(latLon[:,0])/n,np.sum(latLon[:,1])/n,13,'get your own google API key')
    x,y= zip(*[(latLon[0],latLon[1])])
    
    markerList = list(markers.MarkerStyle.markers.keys())
    colorList= np.array()
    normClu = colors.Normalize(np.min(cluUnique),np.max(cluUnique))
    for iClu in cluUnique:
        objLabel = np.where(clu == iClu)
        gmap.scatter(
                latLon[objLabel,0], latLon[objLabel,1],'')
    gmap.draw(writeName)
    
# plots coordinates in 3d with z being the third axis
def plot3d(latLon,z,clu=None):
    n = latLon.shape[0]
    zName=z[0]
    z=z[1]
    #ax.scatter3D(latLon[:,[0]].ravel(),latLon[:,[1]].ravel(),z.ravel())
    if clu is None:
        clu = np.ones(n)
    cluUnique = np.array(np.unique(clu), dtype=int)
    # BTW clu= clusters
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    markerList = list(markers.MarkerStyle.markers.keys())
    normClu = colors.Normalize(np.min(cluUnique),np.max(cluUnique))
    for iClu in cluUnique:
        objLabel = np.where(clu == iClu)
        imClu = ax.scatter3D(
                latLon[objLabel,0].ravel(), latLon[objLabel,1].ravel(),z[objLabel],
                marker=markerList[iClu % len(markerList)],
                c=clu[objLabel], norm=normClu, label=iClu)
    plt.colorbar(imClu)
    plt.legend().set_draggable(True)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel(zName)
    plt.title(f'Position vs {zName} for all boats')
    
# clu is the VID for the 1 ship we wish to plot, zOnXAxis allows you to rotate the graph easily
def plot1ship(latLon,z,VID,clu=None,zOnXAxis=False):
    fig = plt.figure()
    zName=z[0]
    z=z[1]
    ax = fig.gca(projection='3d')
    objLabel = np.where(clu == VID)
    if zOnXAxis:
        ax.scatter3D(
            z[objLabel], latLon[objLabel,0].ravel(),latLon[objLabel,1].ravel())
        ax.set_xlabel(zName)
        ax.set_ylabel('Longitude')
        ax.set_zlabel('Latitude')
    else: 
        ax.scatter3D(
            latLon[objLabel,0].ravel(), latLon[objLabel,1].ravel(),z[objLabel])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel(zName)
    plt.title(f'Position vs {zName} for VID: {VID}')
    plt.legend().set_draggable(True)

# more generic function to plot any features x,y such that x= ["description", data array]
def plotxy1ship(x,y,VID,clu=None):
    plt.figure()
    objLabel = np.where(clu == VID)
    plt.xlabel(x[0])
    plt.ylabel(y[0])
    plt.title(f'{x[0]} vs {y[0]} for VID: {VID}')
    x=x[1]
    y=y[1]
    plt.scatter(x[objLabel],y[objLabel])
