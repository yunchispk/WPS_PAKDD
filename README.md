# WPS (PAKDD 2023)
WPS: An Effective WGAN-based Anomaly Detection Model for loT Multivariate Time Series 

Published on Pacific-Asia Conference on Knowledge Discovery and Data Mining, PAKDD 2023: Advances in Knowledge Discovery and Data Mining, pp 80–91 (https://link.springer.com/chapter/10.1007/978-3-031-33374-3_7)

Motivation

- the GANs usually suffer from training instability caused by vanishing gradient and mode collapse

- anomaly detection stil challenging subtle anomalies are in

Contributions

- Our proposed method using Wasserstein Distance versus GP greatly improves the training stability and mitigates the pattern collapse problem of the GAN-based time series anomaly detection model.

- We use an enhanced co-optimization strategy that combines three kinds of errors to recognize subtle anomalies and reduce the false alarm rate.

- experiments Extensive manifest WPS outperforms the suboptimal model by 17.68% and 10.41% in that precision public datasets. and F1 on three

## Get Started

1. Install Python 3.6, PyTorch >= 1.4.0. 
```
pip install -r requirements.txt
```
2. Download data. You can obtain four benchmarks from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/9605612594f0423f891e/) or [Google Cloud](https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm?usp=sharing). **All the datasets are well pre-processed**. For the SWaT dataset, you can apply for it by following its official tutorial.
3. Train and evaluate. We provide the experiment jupyter notebooks of the mode and all benchmarks named as `./test_*.ipynb`. 

Especially, we use the adjustment operation proposed by [Xu et al, 2018](https://arxiv.org/pdf/1802.03903.pdf) for model evaluation. If you have questions about this, please see this [issue](https://github.com/thuml/Anomaly-Transformer/issues/14).

## Citation
If you find this repo useful, please cite our paper. 

```
@inproceedings{qi2023effective,
  title={An Effective WGAN-Based Anomaly Detection Model for IoT Multivariate Time Series},
  author={Qi, Sibo and Chen, Juan and Chen, Peng and Wen, Peian and Shan, Wenyu and Xiong, Ling},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={80--91},
  year={2023},
  organization={Springer}
}
```

## Contact
If you have any question, please contact qisibo@stu.xhu.edu.cn.
