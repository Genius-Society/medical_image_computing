# Chapter IV - Medical Signal Segmentation and Classification (MSSC)
[![license](https://img.shields.io/github/license/Genius-Society/medical_image_computing.svg)](https://github.com/Genius-Society/medical_image_computing/blob/mssc/LICENSE)

Medical Signal Segmentation and Classification

## Intro
The project is dedicated to the segmentation and classification of medical imaging signals including boundary detection filters, Kmeans clustering algorithms, support vector machines (SVM) and other methods. The dataset uses MRI data of size 157 x 189 x 68.

## Usage
### Environment
```bash
conda create -n py311 python=3.11 -y
conda activate py311
pip install -r requirements.txt
```

### Code download
```bash
git clone -b mssc git@github.com:Genius-Society/medical_image_computing.git
cd medical_image_computing
```

### Classification
```bash
python Classification.py
```

### Segmentation
```bash
python Segmentation.py
```

## Edge Filters
Edge provides critical information about the shape of the region of interest (ROI) and serves as an important step in many segmentation algorithms. In this task, we work on the data in the axial view at slice 16 for edge detection. 
 
### (1)	Please calculate the image gradients along x and y directions with the function cv2.Sobel(), and the gradient magnitude using Numpy.sqrt() function, and plot image gradients ("Gx", "Gy") as well as gradient magnitude ("Gmat") using imshow() function.
```python
Gx = cv2.Sobel(im_z, cv2.CV_64F, 1, 0, ksize=3)
Gy = cv2.Sobel(im_z, cv2.CV_64F, 0, 1, ksize=3)
Gmat = np.sqrt(Gx**2.0 + Gy**2.0)

plt.subplot(1,3,1), plt.imshow(Gx, cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2), plt.imshow(Gy, cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3), plt.imshow(Gmat, cmap = 'gray')
plt.title('Final'), plt.xticks([]), plt.yticks([])
plt.show()
```

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233114923-a88324fe-1800-45a5-a6d1-1ecc6b0da77d.png"/><br>
<b>Figure 2: Results of task 2(1)</b>
</div>

### (2)	Please define the Prewitt kernels in both x and y directions, and use the cv2.filter2D() function to complete prewitt edge filtering.
```python
kernel_x = np.array([[1, 0, -1],[1, 0, -1],[1, 0, -1]])
kernel_y = np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]])
Gx = cv2.filter2D(im_z, -1, kernel_x)
Gy = cv2.filter2D(im_z, -1, kernel_y)
Gmat = np.sqrt(Gx**2.0 + Gy**2.0)

plt.subplot(1,3,1), plt.imshow(Gx, cmap = 'gray')
plt.title('Prewitt X'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2), plt.imshow(Gy, cmap = 'gray')
plt.title('Prewitt Y'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3), plt.imshow(Gmat, cmap = 'gray')
plt.title('Final'), plt.xticks([]), plt.yticks([])
plt.show()
```

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233114758-3c608c2e-01fd-418a-a031-05ad6a8e4a93.png"/><br>
<b>Figure 3: Results of task 2(2)</b>
</div>

### (3)	Please conducting a Canny edge detection with the function of feature.canny() and plot the results using imshow() function. Please briefly explain Canny edge detection. What are the values and meaning of the double thresholds used in your canny edge detection? Please change the lower and upper thresholds to (2, 5) and (3, 15), respectively; How does the edge detection change?
```python
ef1 = feature.canny(im_z, 1.0, 2, 5)
ef2 = feature.canny(im_z, 1.0, 3, 15)

plt.subplot(1,2,1), plt.imshow(ef1, cmap = 'gray')
plt.title('(2, 5)'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2), plt.imshow(ef2, cmap = 'gray')
plt.title('(3, 15)'), plt.xticks([]), plt.yticks([])
plt.show()
```

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233114820-d2a3e181-dc89-4138-9518-4d9a36df8ac3.png"/><br>
<b>Figure 4: Results of task 2(3)</b>
</div>

### (4)	What is spatial frequency? Are edge filters low-pass filtering operation or high-pass filtering operation?
The spatial frequency is an independent variable to describe the characteristics of an image. It can decompose the spatial change of an image pixel value into a linear superposition of simple vibration functions with different amplitudes, spatial frequencies and phases. The composition and distribution of this spatial frequency component is called the spatial frequency spectrum. Edge filters belong to high-pass filtering operation.

## Kmeans clustering
### (1)	Please describe the process of k-means clustering. What is the difference between supervised and unsupervised methods? What are their advantages and disadvantages?
Basic steps of k-means algorithm:<br>
a)	Select k objects from the data as the initial cluster centres;<br>
b)	Calculate the distance of each cluster object to the cluster centre to divide;<br>
c)	Calculate each cluster centre again;<br>
d)	Calculate the standard measurement function. If the method reaches the maximum number of iterations, stop, otherwise, continue the operation.

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233115171-01852440-e872-4e67-83a5-fd58b42ed7c5.png"/><br>
<b>Figure 5: Process of K-means clustering</b>
</div><br>

<div align=center><b>Table 1: Differences between supervised and unsupervised methods</b><br></div>

| ML technique             | Supervised                                                                                      | Unsupervised                                                                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Process                  | In a supervised learning model, input and output variables will be given;                       | In unsupervised learning model, only input data will be given;                                                                             |
| Input Data               | Algorithms are trained using labelled data;                                                     | Algorithms are used against data which is not labelled;                                                                                    |
| Algorithms Used          | SVM, NN, Linear and logistics regression, random forest, and Classification trees;              | Unsupervised algorithms can be divided into different categories: like Cluster algorithms, Kmeans, Hierarchical clustering, etc;           |
| Computational Complexity | Supervised learning is a simpler method;                                                        | Unsupervised learning is computationally complex;                                                                                          |
| Use of Data              | Supervised learning model uses training data to learn a link between the input and the outputs; | Unsupervised learning does not use output data;                                                                                            |
| Accuracy of Results      | Highly accurate and trustworthy method;                                                         | Less accurate and trustworthy method;                                                                                                      |
| Real Time Learning       | Learning method takes place offline;                                                            | Learning method takes place in real time;                                                                                                  |
| Number of Classes        | Number of classes is known;                                                                     | Number of classes is not known;                                                                                                            |
| Main Drawback            | Classifying big data can be a real challenge in Supervised Learning;                            | You cannot get precise information regarding data sorting, and the output as data used in unsupervised learning is labelled and not known. |

<div align=center><b>Table 2: Advantages and disadvantages of supervised and unsupervised methods</b><br></div>

| Method        | Supervised                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Unsupervised                                                                                                                                                                                                      |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Advantages    | Analyst has control over the classification; Processing is tied to specific areas of known identity; Errors can be detected and often rectified;                                                                                                                                                                                                                                                                                                                    | No extensive prior knowledge of the study area is required; Opportunity for human error is minimised; Unique classes are recognised as distinct units;                                                            |
| Disadvantages | Analyst imposes a structure on data, which may not match reality; Training classes are generally based on field identification and not on spectral properties hence spectral signatures are forced; Training data selected by the analyst may not be representative of conditions present throughout the image; Training data can be timeconsuming and costly; Unable to recognise and represent special or unique categories not represented in the training data; | Spectral classes are not necessarily information classes; Analyst has little control over classes; Spectral properties change over time hence detailed spectral knowledge of different features may be necessary. |

### (2)	Import KMeans from the sklearn.cluster package.
```python
from sklearn.cluster import KMeans
```

### (3)	Please group the pixels in slice 16 using the function of KMeans(). Show the results with 4, 8, 20 clusters, respectively. To show the results, please replace the intensity value at each pixel with the intensity value of its corresponding cluster centres and display the resulting image use the function imshow(). This will give each cluster a unique colour. Please compare the results.
```python
count = 1
X = im_z.reshape((-1, 1))

estimators = [
    ('kmeans_4', KMeans(n_clusters=4)), 
    ('kmeans_8', KMeans(n_clusters=8)), 
    ('kmeans_20', KMeans(n_clusters=20))
]

for name, est in estimators:   
    kmeans = est.fit(X)
    labels = kmeans.labels_
    choices = kmeans.cluster_centers_.squeeze()
    img = np.choose(labels, choices)
    img.shape = im_z.shape
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, count), plt.imshow(img, plt.cm.Spectral)
    plt.title(name)  
    count += 1
    plt.show()
```

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233115333-a865dc0d-920e-4c76-aee5-0ad7ede5353b.png"/><br>
<b>Figure 6: Results of task 3(3)</b>
</div>

### (4)	If you repeat the algorithm for several times, will the results change?
The results slightly change after the algorithm is repeated for several times:

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233115412-d54e39fc-fc69-41be-887c-dbc6d13c71e2.png"/><br>
<b>Figure 7: Results of task 3(4)</b>
</div>

### (5)	If 4 clusters are to be generated, please plot the relationship between withincluster sums of point-to-centroid distances (kmeans.inertia) and number of iterations (kmeans.n_iter). The x-axis corresponds to the number of iterations and the y-axis corresponds to the within-cluster sums.
```python
trytime = 500
x_list = []
y_list = []

for i in range(trytime):
    kmeans = KMeans(n_clusters=4).fit(im_z)
    x_list.append(kmeans.n_iter_)
    y_list.append(kmeans.inertia_)

ax = plt.gca()
ax.set_xlabel('Number of iterations')
ax.set_ylabel('Within-cluster sums')
ax.scatter(x_list, y_list, c='r', s=20, alpha=0.5)
plt.show()
```

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233115505-6bb4f3dc-2c0e-465b-b424-b741196def4b.png"/><br>
<b>Figure 8: Results of task 3(5)</b>
</div>

## MRI Dataset
The MRI data, D, is stored as a 157-by-189-by-68 NumPy array. You can show image of each of the frame within the dataset using the imshow() function imported from matplotlib.pyplot. Please use this function to show the data in the axial view at slice 16, the data in the sagittal view at slice 64, and the data in the coronal view at slice 64. And set the aspect of the axis scaling (the ratio of y-unit to x-unit) to 0.5 when plotting the images.

```python
import numpy as np
from skimage import io
from skimage import feature
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt

D = io.imread("data/attention-mri.tif")
print(D.shape)

im_x = D[63,:,:]    #sagittal(x)
im_y = D[:,63,:]    #coronal(y)
im_z = D[:,:,15]    #transaxial(z)

plt.subplot(1,3,1), plt.imshow(im_x, cmap = 'gray', aspect = 0.5)
plt.title('Sagittal'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2), plt.imshow(im_y, cmap = 'gray', aspect = 0.5)
plt.title('Coronal'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3), plt.imshow(im_z, cmap = 'gray')
plt.title('Axial'), plt.xticks([]), plt.yticks([])
plt.show()
```

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233115738-4e752119-f149-4a33-b5d9-d3363e5f139e.png"/><br>
<b>Figure 1: Results of task 1</b>
</div>

## Support Vector Machine
Support-vector machines (SVMs) are supervised learning machine learning models widely used for classification and regression tasks. In medical research, SVMs can be used to predict the health status of a patient for a target disease. In this experiment, we are going to train an SVM model to predict Diabetes. Please download the pima.csv file from Canvas and save it to the same directory as this Jupyter notebook and run the following code to load the datasets.

```python
import os
from csv import reader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC

def main():
    # Load data
    with open("data/pima.csv") as f:
        csv_data = reader(f, delimiter=',')
        raw_data = np.array(list(csv_data))

    # Preprocess data
    data_x = []
    data_y = []
    tuple_len = len(raw_data[0])
    for i in raw_data:
        if not i:
            continue
        data_x.append([float(j) for j in i[0:tuple_len - 2]])
        if i[tuple_len - 1] == "yes":
            data_y.append(1)
        else:
            data_y.append(0)

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=73)

    #TODO:
    clf = LinearSVC(loss="hinge", random_state=42).fit(x_train, y_train)
    print(clf.score(x_test, y_test))

if __name__ == "__main__":
    main()
```

```txt
Output: 0.6771653543307087
```