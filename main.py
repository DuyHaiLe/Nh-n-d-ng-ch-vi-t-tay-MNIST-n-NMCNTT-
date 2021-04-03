import matplotlib.pyplot as plt
import os
import numpy as np
import gzip
import statistics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import math
import time

def load_mnist(path, kind = 'train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
    
    with gzip.open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        labels = np.frombuffer(buffer, dtype = np.uint8)
       
    with gzip.open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        buffer = imgpath.read()
        images = np.frombuffer(buffer, dtype = np.uint8).reshape(len(labels), 28, 28).astype(np.int16)
        
    return images, labels
    
def display(X, y):
    fig, ax = plt.subplots(nrows = 2, ncols = 5, sharex = True, sharey = True, )
    ax = ax.flatten()

    for i in range(10):
        img = X[y == i][0]
        ax[i].imshow(img, cmap = 'Greys', interpolation = 'nearest')
        
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    
def myVectorize(arr):
    brr = arr
    brr = np.ndarray.flatten(brr)
    return brr

def avg(arr):
    total = 0
    for value in arr:
        total += value
    return total / len(arr)

def dif(arr):
    return max(arr) - min(arr)

def sampling(arr, sz = 2, func = min):    
    brr = []
    for i in range(0, 28, sz):
        for j in range(0, 28, sz):
            subSquareValues = []
            for ii in range(i, i + sz):
                for jj in range(j, j + sz):
                    subSquareValues.append(arr[ii][jj])
            brr.append(func(subSquareValues))
    
    return np.array(brr);

def hist(arr):
    Bin=np.ndarray(shape=(256),dtype=np.int16)
    for i in range(256):
        Bin[i]=i
        
    res = np.histogram(arr,bins=Bin)
            
    return res[0]

def getAccuracyKNN(Xtrain, ytrain, Xtest, ytest):
    knn = KNeighborsClassifier(n_neighbors = 250)
    knn.fit(Xtrain, ytrain)

    yPred = knn.predict(Xtest)
    return metrics.accuracy_score(ytest, yPred)

def createFeature(X, y):
    ans = np.zeros((10, X.shape[1]), dtype=float)
    cnt = np.zeros((10), dtype=float)
    
    for i in range(0, X.shape[0]):
        cnt[y[i]] += 1
        for j in range(0, X.shape[1]):
            ans[y[i]][j] += X[i][j]
    
    for i in range(0, 10):
        for j in range(0, X.shape[1]):
            ans[i][j] /= cnt[i]
    
    return ans

def euclidDist(X, Y):
    ans = 0.0
    for i in range(X.shape[0]):
        ans += (X[i]-Y[i])*(X[i]-Y[i])

    ans = math.sqrt(ans)
    return ans

def getAccuracyAvg(X_train, y_train, X_test, y_test):
    feature = createFeature(X_train, y_train)
    
    y_pred = []
    
    for x in X_test:
        tmp = 0
        minDist = euclidDist(x, feature[0])
        
        for i in range(1, 10):
            foo = euclidDist(x, feature[i])
            if(foo < minDist):
                minDist = foo
                tmp = i
                
        y_pred.append(tmp)
        
    return metrics.accuracy_score(np.array(y_pred), y_test)

X_train, y_train = load_mnist('data/', kind = 'train')
X_test, y_test = load_mnist(path = 'data/', kind = 'test')

display(X_train, y_train)
display(X_test, y_test)

print('===== BEGIN KNN =====')
print('=== Vectorize ===')

X_train_vect = np.array([myVectorize(x) for x in X_train])
X_test_vect = np.array([myVectorize(x) for x in X_test])

start_time = time.time();
print(getAccuracyKNN(X_train_vect, y_train, X_test_vect, y_test))
print("--- %s seconds ---" % (time.time() - start_time))
'''
start_time = time.time();
print(getAccuracyAvg(X_train_vect, y_train, X_test_vect, y_test))
print("--- %s seconds ---" % (time.time() - start_time))
'''

print("=== Sampling 2, Min ===")
X_train_samp = np.array([sampling(x, 2, min) for x in X_train])
X_test_samp = np.array([sampling(x, 2, min) for x in X_test])

start_time = time.time();
print(getAccuracyKNN(X_train_samp, y_train, X_test_samp, y_test))
print("--- %s seconds ---" % (time.time() - start_time))

print("=== Sampling 4, Min ===")
X_train_samp = np.array([sampling(x, 4, min) for x in X_train])
X_test_samp = np.array([sampling(x, 4, min) for x in X_test])

start_time = time.time();
print(getAccuracyKNN(X_train_samp, y_train, X_test_samp, y_test))
print("--- %s seconds ---" % (time.time() - start_time))

print("=== Sampling 2, Avg ===")
X_train_samp = np.array([sampling(x, 2, avg) for x in X_train])
X_test_samp = np.array([sampling(x, 2, avg) for x in X_test])

start_time = time.time();
print(getAccuracyKNN(X_train_samp, y_train, X_test_samp, y_test))
print("--- %s seconds ---" % (time.time() - start_time))

print("=== Sampling 4, Avg ===")
X_train_samp = np.array([sampling(x, 4, avg) for x in X_train])
X_test_samp = np.array([sampling(x, 4, avg) for x in X_test])

start_time = time.time();
print(getAccuracyKNN(X_train_samp, y_train, X_test_samp, y_test))
print("--- %s seconds ---" % (time.time() - start_time))

print('=== Histogram ===')
X_train_hist = np.array([hist(x) for x in X_train])
X_test_hist = np.array([hist(x) for x in X_test])

start_time = time.time();
print(getAccuracyKNN(X_train_hist, y_train, X_test_hist, y_test))
print("--- %s seconds ---" % (time.time() - start_time))
print('===== END KNN =====')

print('===== BEGIN FEATURE =====')
print('=== Vectorize ===')

X_train_vect = np.array([myVectorize(x) for x in X_train])
X_test_vect = np.array([myVectorize(x) for x in X_test])
'''
start_time = time.time();
print(getAccuracyKNN(X_train_vect, y_train, X_test_vect, y_test))
print("--- %s seconds ---" % (time.time() - start_time))
'''
start_time = time.time();
print(getAccuracyAvg(X_train_vect, y_train, X_test_vect, y_test))
print("--- %s seconds ---" % (time.time() - start_time))


print("=== Sampling 2, Min ===")
X_train_samp = np.array([sampling(x, 2, min) for x in X_train])
X_test_samp = np.array([sampling(x, 2, min) for x in X_test])

start_time = time.time();
print(getAccuracyAvg(X_train_samp, y_train, X_test_samp, y_test))
print("--- %s seconds ---" % (time.time() - start_time))

print("=== Sampling 4, Min ===")
X_train_samp = np.array([sampling(x, 4, min) for x in X_train])
X_test_samp = np.array([sampling(x, 4, min) for x in X_test])

start_time = time.time();
print(getAccuracyAvg(X_train_samp, y_train, X_test_samp, y_test))
print("--- %s seconds ---" % (time.time() - start_time))

print("=== Sampling 2, Avg ===")
X_train_samp = np.array([sampling(x, 2, avg) for x in X_train])
X_test_samp = np.array([sampling(x, 2, avg) for x in X_test])

start_time = time.time();
print(getAccuracyAvg(X_train_samp, y_train, X_test_samp, y_test))
print("--- %s seconds ---" % (time.time() - start_time))

print("=== Sampling 4, Avg ===")
X_train_samp = np.array([sampling(x, 4, avg) for x in X_train])
X_test_samp = np.array([sampling(x, 4, avg) for x in X_test])

start_time = time.time();
print(getAccuracyAvg(X_train_samp, y_train, X_test_samp, y_test))
print("--- %s seconds ---" % (time.time() - start_time))

print('=== Histogram ===')
X_train_hist = np.array([hist(x) for x in X_train])
X_test_hist = np.array([hist(x) for x in X_test])

start_time = time.time();
print(getAccuracyAvg(X_train_hist, y_train, X_test_hist, y_test))
print("--- %s seconds ---" % (time.time() - start_time))
print('===== END FEATURE =====')




