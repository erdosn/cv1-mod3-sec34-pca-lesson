
### Questions
- What are eigenvectors and eigenvalues?
- If you don't have an issue with dimensionality, does PCA improve scores?
- Why the covariance matrix? 
    - The first vector explains the covariance the most and therefore, projecting onto it, maintain the most information in regards to covariance. 
    - Covariance is square, it's symmetrical
        - our eigenvalues get upgraded

- How do you interpret a model when you use PCA? - You can't

### Objectives
- Define Eigenvalues and Eigenvectors
- Describe how these are used in PCA
- Apply PCA to reduce dimensions of data

### Outline
- Questions
- Explain eigenvalues and eigenvectors and why they're awesome
- Apply eigen decomposition to the correlation matrix and discuss how it's used in PCA
- Apply PCA to some dataset that we create using sklearn


```python
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
```

### Eigenvalues and Eigenvectors are only applied on square matrices


```python
A = np.random.randint(10, 20, size=(3, 3))
A 
```




    array([[13, 11, 17],
           [13, 11, 15],
           [15, 19, 13]])



### Eigen Properties
- Eigen Values multiply to the determinent of A
- Eigen Values add to the trace of matrix A (trace = sum of diagonals)
- Eigen Vectors are orthonormal


```python
evals, evecs = np.linalg.eig(A)
```


```python
evals_diag = np.diag(evals)
evals_diag
```




    array([[42.46064274,  0.        ,  0.        ],
           [ 0.        , -0.834997  ,  0.        ],
           [ 0.        ,  0.        , -4.62564573]])




```python
evals, evecs
```




    (array([42.46064274, -0.834997  , -4.62564573]),
     array([[-0.56327554, -0.82567166, -0.58550079],
            [-0.53355856,  0.30738993, -0.25241239],
            [-0.63090089,  0.47305152,  0.77037446]]))




```python
np.prod(evals), np.linalg.det(A)
```




    (164.00000000000043, 164.00000000000006)




```python
np.sum(evals), np.trace(A)
```




    (37.0, 37)



### Eigenvalues


```python
evals.prod(), np.linalg.det(A)
```




    (-398.0000000000008, -398.0)




```python
np.trace(A), np.sum(evals)
```




    (41, 40.99999999999996)



### Eigenvectors if A is symmetrical


```python
### Let's make a symmetrical matrix
# A Matrix M is symmetrical iff M.T == M
A  = np.random.randint(10, 100, size=(5000, 3))
A_sym = np.cov(A.T) 
A_sym
```




    array([[681.28476795,   2.3989822 ,  -9.3305183 ],
           [  2.3989822 , 678.87066629,   1.51813915],
           [ -9.3305183 ,   1.51813915, 681.83669978]])




```python
evals, evecs = np.linalg.eig(A_sym)
evals, evecs
```




    (array([690.92441379, 671.2152533 , 679.85246694]),
     array([[-0.70101091, -0.68125763,  0.21088326],
            [-0.04991863,  0.34185566,  0.93842572],
            [ 0.71140132, -0.64731967,  0.27365199]]))




```python
### normal -> all eigenvectors are normal, always
# A vector v is normal iff length of v is 1
np.sqrt(np.sum(evecs[:, 0]**2))
```




    0.9999999999999999




```python
### because A_sym is symmetrical the vectors are also orthogonal
# vectors a and b are orthogonal iff the angle between a and b is 90 degree (dot product = 0)

np.dot(evecs[:, 0], evecs[:, 1])
```




    2.706168622523819e-15




```python
## if A*B = I -> A, B are inverses
## evecs.T = evecs.inv()
np.round(evecs.dot(evecs.T), 1)
```




    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])




```python
### eigenvecs and vals redescribe your space 
### if your space is symmetrical, then the eigenvecs are a basis
### so...why is this important for dimensionality reduction?
```

## PCA
1. Calculate the covariance matrix of your data, C -> Symmetrical
2. Calculate the eigenvecs and eigenvalues of covariance matrix
3. Project our data onto the vectors that most describe the correlation


```python
A_sym
```




    array([[681.28476795,   2.3989822 ,  -9.3305183 ],
           [  2.3989822 , 678.87066629,   1.51813915],
           [ -9.3305183 ,   1.51813915, 681.83669978]])




```python

```




    array([[ 339.53151684,   22.8263153 ,  101.99325328],
           [  23.49657348,   78.44176611, -412.98415117],
           [ 103.65429583, -407.73737703,   50.91103213]])



### Decomposition for non square matrix
- SVD (Singular Value Decomposition)
- Reduces any matrix

### Assessment
- I learned how information is being maintained as data is moved into lower dimensional space
- I now understand that after (eigen decomposition) calculating eigens, we can project the data into four quadrants based upon eigen V1 and V2
- The principal component.  Comes from the covariance matrix. The vector that explains the most variance and retains the most of the original data info
- I definitely need a lot more time with this topic. I did learn that PCA is useful for finding relatedness between populations.
