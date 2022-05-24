# DA-Net: Dual-attention network for multivariate time series classification



**Abstract:** Multivariate time series classification is one of the increasingly important issues in machine learning. Existing multivariate time series methods focus on establishing global long-range dependencies or discovering critical local sequence fragments. However, the combined information from them are essential to make classifying time series more effective. It is challenging to mine critical local sequence fragments to establish global long-range dependencies. This paper proposes a Dual Attention Network (DA-Net) to mine the local-global features for multivariate time series classification. Specially, the proposed DA-Net consists of two distinctive layers, Squeeze-Excitation Window Attention (SEWA) layer and Sparse Self-Attention within Windows (SSAW) layer.  For the SEWA layer, we capture local window-wise information by explicitly establishing window dependencies to prioritize critical windows. For the SSAW layer, we preserve rich activate scores with less computation to widen the window scope and thus capture global long-range dependencies. Based on the two elaborated layers, DA-Net can mine critical local sequence fragments in the process of establishing global long-range dependencies. Specifically, the proposed DA-Net can mine local features of multivariate time series fragment windows and establish global long-range dependencies.  In addition, we extend our methodology with its variants on the 30 public UEA datasets for multivariate time series classification. The experimental results exhibit that DA-Net achieves competing performances with state-of-the-art approaches, yields a strong baseline and gaps the bridge for Transformer networks on multivariate time series classification. 

## Requirements

* Python 3.6
* PyTorch version 1.8.2+cu111
* einops
* seaborn
* sklearn


## Run Model Training and Evaluation

## Datasets

To save time, we recommend to download the FaceDetection dataset from the [here ]([https://](https://drive.google.com/file/d/1-XnZ-ZEqoaoJCOqSnFHMZiTYmKuOQGoa/view?usp=sharing)). The overall 30 UEA datasets can be download from [here](http://www.timeseriesclassification.com).

- Create the Folder 'data'.
- unzip the 'FaceDetection' on the data folder.

**Train/Test**:

```bash
python run_UEA.py
```

**Visualization**:

```
python draw_picture/plot_TSNE.py model_path=<saved model path>
```

