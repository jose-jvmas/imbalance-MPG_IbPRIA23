# Addressing class imbalance in Multilabel Prototype Generation for k-Nearest Neighbor classification

## Authors
 Carlos Penarrubia<sup>&dagger;</sup>, Jose J. Valero-Mas<sup>&dagger;,&Dagger;</sup>, Antonio Javier Gallego<sup>&dagger;</sup>, and Jorge Calvo-Zaragoza<sup>&dagger;</sup>
 
&dagger;*University Institute for Computer Research, University of Alicante, Alicante, Spain*

&Dagger;*Music Technology Group, Universitat Pompeu Fabra, Barcelona, Spain*

 
## Description
Extensions to different Multilabel Prototype Generation methods to deal with imbalance data. Work accepted at the *Iberian Conference on Pattern Recognition and Image Analysis (IbPRIA) 2023*.
 

 
## Contents
- *MPG/* : Contains the implementations of the base multilabel Prototype Generation methods together with the proposed extensions. These methods are (together with their original reference):
	- Multilabel Reduction through Homogeneous Clustering (MRHC) [^1]
	- Multilabel Chen (MChen) [^2]
	- Multilabel Reduction through Space Partitioning, version 3 (MRSP3) [^2]
- *Experiments.py* : Main script for performing the experimentation included in the manuscript.
- *Metrics.py*: Class including the evaluation metrics.
- *StatisticalAnalysis.py*: Script prepared for performing the statistical analysis once the results have been obtained.

 [^1]: Ougiaroglou, S., Filippakis, P., & Evangelidis, G. (2021). Prototype generation for multi-label nearest neighbours classification. In: Proceedings of the 16th International Conference on Hybrid Artificial Intelligent Systems, Bilbao, Spain, September 22–24, 2021, (pp. 172-183).
 [^2]: Valero-Mas, J. J., Gallego, A. J., Alonso-Jiménez, P., & Serra, X. (2023). Multilabel Prototype Generation for data reduction in K-Nearest Neighbour classification. Pattern Recognition, 135, 109190.
 
## Usage
For the reproduction of the experiments included in the paper, please proceed as follows:
```
$ pip install -r requirements.txt
$ python Experiments.py
$ python StatisticalAnalysis.py
```
