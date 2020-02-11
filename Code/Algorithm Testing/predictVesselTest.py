# -*- coding: utf-8 -*-
"""
A test program where I test, analyze, and compare algorithms I wanted to use for the solution.
Many lines of code are commented out for testing different things and code blocks most likely contain incomplete thoughts.
Some segments of code might just not run entirely, since I changed an import they depended on to test something else

Requires utils.py file which is used for data unpacking and plotting/analysis 
@author: Brian Blakely
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def predictWithK(testFeatures, numVessels, trainFeatures=None, 
                 trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)
    km = KMeans(n_clusters=numVessels, random_state=100)
    predVessels = km.fit_predict(testFeatures)
    
    return predVessels

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    # Arbitrarily assume 20 vessels
    return predictWithK(testFeatures, 20, trainFeatures, trainLabels)

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    from utils import loadData, plotVesselTracks, plot3d
    data = loadData('set1.csv')
    data2= loadData('set2.csv')
    data3= loadData('set3noVID.csv')
    features2=data2[:,2:]
    labels2=data2[:,1]
    features2[:,4]=features2[:,4]/10
    features = data[:,2:]
    features[:,4]=features[:,4]/10
    labels = data[:,1]
    featureNames=['Time','Latitude','Longitude','Speed (1022 MAX)','Angle of Movement']
    time=features[:,[0]]-50400
    features3=data3[:,2:]
    features[:,0]=time[:,0]# time is now in minutes
    features[:,3]=data[:,5]*(1.852)/12 # speed is now in km per second
    #%% Plot all vessel tracks with no coloring
    plt.ion()
    plotVesselTracks(features[:,[2,1]])
    plot3d(features[:,[2,1]],('Time',features[:,[0]]-50000))
    plt.title('All vessel tracks')
    
    #%% PCA 
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, svd_solver='full')
    pca.fit(features)        
    scaler = StandardScaler()
    clustering = DBSCAN(eps=.1, min_samples=2,algorithm='ball_tree').fit(scaler.fit_transform(features2))
    adjusted_rand_score(labels2, clustering.labels_)
    plotVesselTracks(features2[:,[2,1]], clustering.labels_)
    plt.title('Vessel tracks by cluster with K')
    
    from scipy.fftpack import fft
    y= fft(features)
    #%%
    import trajectory
    trajectory.trajectory.function(features[1],features[18])
    np.sum(abs(features2[0,0]-features[:,0]))
    eq=np.empty(0)
    pred=np.array([10000])
    for i in features2:
        eq=np.array([1000])
        for j in features:
            eq=np.vstack([eq,np.array((abs(i[0]-j[0])*.4+abs( geopy.distance.distance(i[[1,2]],j[[1,2]]).km)*.5+abs(i[3]-j[3])*.3+abs(i[4]-j[4])*.2))])
        pred=np.vstack([pred,labels[np.argmin(eq)]])
        
        
    #%% RF
    from sklearn.ensemble import RandomForestClassifier as rf
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.cluster import SpectralClustering as sc
    from sklearn.utils import resample
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(features)
    clf = rf(n_estimators=400,max_depth=8, random_state=1)
    clf.fit(features,labels)
    z= clf.predict(features2)
    np.unique(z).shape
    adjusted_rand_score(labels2, z)
    z=resample(features2)
    
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(z)
    pred=KMeans(n_clusters=8,random_state=100).fit_predict(features2)
    adjusted_rand_score(labels2, pred)
    #%% Run prediction algorithms and check accuracy
    
    # Prediction with specified number of vessels
    numVessels = np.unique(labels).size
    predVesselsWithK = predictWithK(features, numVessels)
    ariWithK = adjusted_rand_score(labels, predVesselsWithK)
    
    # Prediction without specified number of vessels
    predVesselsWithoutK = predictWithoutK(features)
    predNumVessels = np.unique(predVesselsWithoutK).size
    ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)
    
    print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
    print(f'Adjusted Rand index for estimated K = {predNumVessels}: '
          + f'{ariWithoutK}')

    #%% Plot vessel tracks colored by prediction and actual labels
    plotVesselTracks(features[:,[2,1]], predVesselsWithK)
    plt.title('Vessel tracks by cluster with K')
    plotVesselTracks(features[:,[2,1]], predVesselsWithoutK)
    plt.title('Vessel tracks by cluster without K')
    plotVesselTracks(features[:,[2,1]], labels)
    plt.title('Vessel tracks by label')
    #%% trajectory testing
    # find points really close to each other within 100 seconds
    # look at next 100 points and find the minimal 

    #%% 3d testing
    import utils
    #shift time to start at time 0
    time=feat[:,[0]]-50400
    feat[:,0]=time[:,0]
    #features[:,[3]]
    feat=np.append(features,features3,axis=0)
    label=np.append(labels,labels2,axis=0)
    plotVesselTracks(feat[:,[2,1]], label)
    plotVesselTracks(features2[:,[2,1]], labels2)
    plotVesselTracks(features3[:,[2,1]])
    utils.plot3d(feat[:,[2,1]],('Time',time), label)
    utils.plot3d(features3[:,[2,1]],(featureNames[0],features3[:,0]))
    utils.plot3d(features2[:,[2,1]],('Time',time), labels2)
    plt.title('Vessel tracks by label')
    utils.plot1ship(features[:,[2,1]],(featureNames[0],time),np.unique(labels)[0],labels,1)
    utils.plot1ship(features[:,[2,1]],(featureNames[4],features[:,4]/10),np.unique(labels)[1],labels,1)
    utils.plotxy1ship((featureNames[0],time),(featureNames[4],features[:,4]/10),np.unique(labels)[1],labels)
# to get lat and long for a specific VID: features[np.where(labels==np.unique(labels)[1])][:,[2,1]]
    midLat=(np.max(features[:,1])+np.min(features[:,1]))/2
    midLong=(np.max(features[:,2])+np.min(features[:,2]))/2
    utils.plotxy1ship(('Lat',features[:,1]-midLat),('Long',features[:,2]-midLong),np.unique(labels)[0],labels)
    #%% Polar Plot Testing
    # need to centralize data and then compute distance r
    objLabel=np.where(np.unique(labels)[0]==labels)
    #np.linalg.norm(features[objLabel,1]-midLat)
    radius=np.hypot(features[objLabel,1]-midLat,features[objLabel,2]-midLong)
    fig=plt.figure()
    ax=fig.gca(polar=True)
    plt.scatter(features[objLabel,4]/10,radius)    

    #%% Average, Max, Min Speeds
    avg=np.zeros(np.shape(np.unique(labels))[0])
    maxi=np.zeros((np.shape(np.unique(labels))[0],2))
    i=0
    for id in np.unique(labels):
        #print(features[np.where(100018.0==labels),3])
        avg[i]=np.sum(features[np.where(id==labels),3])/np.shape(features[np.where(id==labels),3])[1]
        maxi[i]=(np.max(features[np.where(id==labels),3]),np.min(features[np.where(id==labels),3]))
        i+=1
        
    plt.bar(np.unique(labels),avg)
    plt.xticks(np.unique(labels))
    plt.title("Average")
    plt.plot
    
    plt.bar(np.unique(labels),maxi[:,0])
    plt.xticks(np.unique(labels))
    plt.title("Max")
    plt.plot
    
    plt.bar(np.unique(labels),maxi[:,1])
    plt.xticks(np.unique(labels))
    plt.title("Min")
    plt.plot
    #%% NN Unsuper
    import NNunsuper as nn
    foo=nn.predwithNN(features)
    
    adjusted_rand_score(foo[0][:,np.min([np.arange(1,20)])],labels)
    from sklearn.cluster import AgglomerativeClustering as agc
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(features2)
    clust= agc(n_clusters=8).fit(testFeatures)
    adjusted_rand_score(clust.labels_,labels2)
    
    testFeatures=scaler.fit_transform(features2)
    clust=KMeans(n_clusters=8).fit_predict(testFeatures)
    adjusted_rand_score(clust,labels2)
    
    from sklearn.cluster import SpectralClustering as sc
    idk=sc(n_clusters=8,random_state=1)
    clust=idk.fit_predict(testFeatures)      
    adjusted_rand_score(clust,labels2)
    #%% Determining how much a ships lat and long will move relative to speed
    foo= features[np.where(100012.0==labels)[0],:]
    deltaTime=np.zeros(foo.shape[0]-1)
    deltaPos=np.zeros((foo.shape[0]-1,2))
    for i in range(foo.shape[0]-1):
        deltaTime[i]=foo[i+1,0]-foo[i,0]
        deltaPos[i,0]=foo[i+1,1] - foo[i,1] # latitude 
        deltaPos[i,1]=foo[i+1,2] - foo[i,2] #longitude
    print(np.sum(deltaTime)/np.shape(deltaTime))
    print("longitude = x, latitude = y")
    #%%
    from gmplot import gmplot
    uniq=np.array(np.unique(labels),dtype=int)
    z=np.array([['#000000'],['#0000CD'],['#0000FF'],['#00BFFF'],['#00CED1'],['#00FA9A'],['#00FF00'],['#00FF7F']])
    gmap= gmplot.GoogleMapPlotter(36.931199,-76.113555,13,'get your own google API key')
    lat3,long3=zip(*[(features3[:,1],features3[:,2])])
    gmap.scatter(lat3[0],long3[0],size=30,marker=False)
    gmap.draw("F:\Python File Saves\html's\lastSet.html")
    x,y= zip(*[(np.append(features[:,1],features2[:,1]),np.append(features[:,2],features2[:,2]))])
    gmap.scatter(x[0],y[0],z[0],size=30,marker=False)
    gmap.draw("F:\Python File Saves\html's\scatter.html")    
    gmap.heatmap(x[0],y[0])    
    gmap.draw("F:\Python File Saves\html's\heatmap.html")   
    gmap.plot(x[0],y[0])
    gmap.draw("F:\Python File Saves\html's\plot.html")
    
    #%%
    from sklearn.cluster import DBSCAN
    import pandas as pd
    kms_per_radian = 6371.0088
    epsilon = 2 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(features[:,[1,2]]))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series([features[:,[1,2]][cluster_labels == n] for n in range(num_clusters)])
    print('Number of clusters: {}'.format(num_clusters))
    
    #%%
    import geopy.distance
    from dipy.segment.metric import Metric
    from dipy.segment.metric import ResampleFeature
    
    class GPSDistance(Metric):
        def __init__(self):
            super(GPSDistance,self).__init__(feature=ResampleFeature(nb_points=256))
        def are_compatible(self,shape1,shape2):
            return len(shape1)==len(shape2)
        def dist(self,v1,v2):
            x= [geopy.distance.vincenty([p[0][0],p[0][1]],[p[1][0],p[1][1]]).km for p in list(zip(v1,v2))]
            currD=np.mean(x)
            return currD
    
    from dipy.segment.clustering import QuickBundles
    metric= GPSDistance()
    feature=ResampleFeature(nb_points=256)
    qb= QuickBundles(threshold=1,metric=metric)
    qb.cluster(features[:,1],features[:,2])
    clusters=qb.cluster(features[:,0])
    print(f"Nb.clusters: {len(clusters)}")
    
    
    
    #%% Haversine analysis
    from sklearn.cluster import AffinityPropagation as ap
    from sklearn import preprocessing    
    import erp
    
    def haversine(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
    
        All args must be of equal length.    
    
        """
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
        dlon = lon2 - lon1
        dlat = lat2 - lat1
    
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        return km

    #haversine(features[0,1],features[0,2],features[6,1],features[6,2])
    
    def locationPrediction(example,features):
        # predict lat and long based off current angle
        # returns elligable candidates
        time=example[0]
        lat=example[1]
        lon=example[2]
        speed=example[3]
        theta=example[4]
        print(theta)
        if example[3]==0:
            print("is 0")
            return np.where((features[:,[2,1]]-[lat,lon]<.02) & (speed<5) )
        if theta<90:
            #first quadrant
            print("1st quad")
            return np.where((features[:,1]>lat) & (features[:,2]>lon) & (abs(time-features[:,0])<260))
        elif theta<180:
            #2nd 
            print("2nd quad")
            return np.where((features[:,1]>lat) & (features[:,2]<lon) & (abs(time-features[:,0])<260))
        elif theta<270:
            #3rd
            print("3rd quad")
            return np.where((features[:,1]<lat) & (features[:,2]<lon) & (abs(time-features[:,0])<260))
        else:
            #4th
            print("4th quad")
            return np.where((features[:,1]<lat) & (features[:,2]>lon)& (abs(features[:,3]-speed)<100) & (abs(time-features[:,0])<260))
    test=locationPrediction(foo[0,:].reshape(-1,1),features)
    smallest=None
    for i in range(np.shape(test[0])[0]):
        h=haversine(foo[0,1],foo[0,2],features[test[0]][i,1],features[test[0]][i,2])
        if smallest==None:
            smallest=h
        elif h<smallest:
            smallest=h
            print(i)
    plt.scatter(features[test[0]],features[test[0]])
    plt.plot(foo[0,[2,1]],foo[1,[2,1]])
    for i in range(10):
        print(haversine(foo[i,1],foo[i,2],foo[i+1,1],foo[i+1,2]))
        
    
   # erp.e_erp(features[1,[1,2,3,4]].reshape(-1,1),features[2,[1,2,3,4]].reshape(-1,1),0)
    plotVesselTracks(features[:,[2,1]])
    utils.plot1ship(features[:,[2,1]],(featureNames[0],time),100007,labels)
    
    #test=ap().fit_predict(norm)
    #adjusted_rand_score(test,labels)
    plt.scatter(features[:,1],features[:,2],s=7)
    plt.scatter(features2[:,1],features2[:,2],s=7)
    utils.plot3d(features2[:,[2,1]],('Time',features2[:,0]))
    
