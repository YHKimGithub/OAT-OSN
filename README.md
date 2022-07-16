# A Sliding Window Scheme for Online Temporal Action Localization (OAT-OSN)   
**A Sliding Window Scheme for Online Temporal Action Localization** (ECCV2022)   
Young Hwi Kim, Hyolim Kang, Seon Joo Kim   
[[`link`]()]   

## Updates
**17 Jul, 2022**: Initial release

## Installation

### Prerequisites
- Ubuntu 18.04  
- Python 3.8.8   
- CUDA 11.0  

### Requirements
- pytorch==1.8.1  
- numpy==1.19.2
- h5py==3.6.0
- ...

To install all required libraries, execute the pip command below.
```
pip install -r requirement.txt
```

## Training

### Input Features
We provide the Kinetics pre-trained feature of THUMOS'14 dataset.
The extracted features can be downloaded from [link will be added soon].   
Files should be located in 'data/'.  
You can also get the feature files from [here](https://github.com/wangxiang1230/OadTR).

### Trained Model
The trained models that used Kinetics pre-trained feature can be downloaded from [link will be added soon].    
Files should be located in 'checkpoints/'. 

### Training Model by own
To train the main OAT model, execute the command below.
```
python main.py --mode=train
```
To train the post-processing network (OSN), execute the commands below.
```
python supnet.py --mode=make --inference_subset=train
python supnet.py --mode=make --inference_subset=test
python supnet.py --mode=train
```


## Testing
To test OAT-OSN, execute the command below.
```
python main.py --mode=test
```

To test OAT-NMS, execute the command below.
```
python main.py --mode=test --pptype=nms
```

## Results

| Method | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:| 
| OAT-OSN | 63.0 | 56.7 | 47.1 | 36.3 | 20.0 |
| OAT-NMS | 69.7 | 64.0 | 53.9 | 42.9 | 27.0 |


## Citing OAT-OSN
Please cite our paper in your publications if it helps your research:

```BibTeX
TBD
```
