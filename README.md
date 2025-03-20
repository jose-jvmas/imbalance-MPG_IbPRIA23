# Addressing class imbalance in Multilabel Prototype Generation for k-Nearest Neighbor classification

**Carlos Penarrubia<sup>&dagger;</sup>, Jose J. Valero-Mas<sup>&dagger;,&Dagger;</sup>, Antonio Javier Gallego<sup>&dagger;</sup>, and Jorge Calvo-Zaragoza<sup>&dagger;</sup>**
 
&dagger;*University Institute for Computer Research, University of Alicante, Alicante, Spain*

&Dagger;*Music Technology Group, Universitat Pompeu Fabra, Barcelona, Spain*

<pre>
@inproceedings{Penarrubia2023,
	author = {Penarrubia, Carlos and Valero-Mas, Jose J. and Gallego, Antonio Javier and Calvo-Zaragoza, Jorge},
	title = {Addressing Class Imbalance in Multilabel Prototype Generation for k-Nearest Neighbor Classification},
	booktitle = {Pattern Recognition and Image Analysis: 11th Iberian Conference, IbPRIA 2023, Alicante, Spain, June 27–30, 2023, Proceedings},
	pages = {15–27},
	location = {Alicante, Spain},
	year = {2023},
	isbn = {978-3-031-36615-4},
	publisher = {Springer-Verlag},
	address = {Berlin, Heidelberg},
	doi = {10.1007/978-3-031-36616-1_2}
}
</pre>

 
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


## Acknowledgments

This work was supported by the I+D+i project TED2021-132103A-I00 (DOREMI), funded by MCIN/AEI/10.13039/501100011033.

<a href="https://www.ciencia.gob.es/" target="_blank"><img src="https://www.dlsi.ua.es/~jgallego/projects/DOReMI/images/logo_min.png" style="height:100px" alt="Ministerio de Ciencia e Innovación"></a> 
&nbsp;
<a href="https://commission.europa.eu/strategy-and-policy/recovery-plan-europe_es" target="_blank"><img src="https://www.dlsi.ua.es/~jgallego/projects/DOReMI/images/logo_ue.png" style="height:100px" alt="Financiado por la Unión Europea, fondos NextGenerationEU"></a>
<br>
<a href="https://planderecuperacion.gob.es/" target="_blank"><img src="https://www.dlsi.ua.es/~jgallego/projects/DOReMI/images/logo_plan_recuperacion_transformacion_resiliencia.png" style="height:100px" alt="Plan de Recuperación, Transformación y Resiliencia"></a>
&nbsp;
<a href="https://www.aei.gob.es/" target="_blank"><img src="https://www.dlsi.ua.es/~jgallego/projects/DOReMI/images/logo_aei.png" style="height:100px" alt="Agencia Estatal de Investigación"></a>

<br/>

