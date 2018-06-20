# DDPN

This project is the implementation of the paper **Rethinking Diversified and Discriminative Proposal Generation for Visual Grounding**.The network architecture with DDPN for our visual grounding model is illustrated in Figure 1.

<img src="https://github.com/XiangChenchao/DDPN/raw/master/images/DDPN.jpg" alt="Figure 1: The model architecture for our visual grounding model." width="60%"/>
<center>Figure 1: The model network architecture for our visual grounding model.</center>


## Requirements
- Python version 2.7
- easydict 
- cv2
- Pytorch 0.3 (optional, used for speed-up multi-threads data loading)

## Pretrained Models

We release the trained models on four datasets, which achieve slightly better results than that shown in the paper. 

|   Datasets    | Flickr30k-Entities | Referit | Refcoco  | Refcoco+ |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| val   | 72.65   | 63.63%  | 77.18%  | 65.12%  |
| test  | 73.34   | 63.50%  | 76.70%  | 63.65%  |
| testA |         |         | 80.57%  | 70.54%  |
| testB |         |         | 72.42%  | 55.59%  |

1. Download pretrained models [BaiduYun](https://pan.baidu.com/s/1QiLu28UoOePCHe2W_2gNVA)
2. Unzip the model files in directory './pretrained_model'.


## Preprocess
- 1. Caffe
  ```
  cd ./caffe
  make all -j32
  make pycaffe
  ```

- 2. Download Images, **Images only**
  - **flickr30k-entities** 
    - download the [Flickr30k-Entities images](https://drive.google.com/file/d/0B_PL6p-5reUAZEM4MmRQQ2VVSlk/view?usp=sharing)
    - move flickr30k-entities images to directory './data/flickr30k/flickr30k-images/'.
  - **referit**, download the Referit Images.
    ```
    wget -O ./data/referit/ImageCLEF/referitdata.tar.gz http://www.eecs.berkeley.edu/~ronghang/projects/cvpr16_text_obj_retrieval/referitdata.tar.gz
    tar -xzvf ./data/referit/ImageCLEF/referitdata.tar.gz -C ./data/referit/ImageCLEF/
    ```
  - **refcoco/refcoco+**, download the **mscoco train2014 Images**
    - [mscoco train2014](http://images.cocodataset.org/zips/train2014.zip).
    - move images of mscoco train2014 to directory './data/mscoco/image2014/train2014/'

- 3. Extract [DDPN image features](https://github.com/yuzcccc/bottom-up-attention#extract-features). For a 3xhxw image, we extract the 2048-D visual feature and 4-D spatial feature (post-processed to 5-D) as the input feature for our model. The script we use is as follows. Note that we use **--num_bbox 100,100** to extract a fix number of proposals (K=100) for each image. 
  ```
  ./tools/extract_feat.py --gpu 0,1,2,3 --cfg experiments/cfgs/faster_rcnn_end2end_resnet_vg.yml --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt --net /path/to/caffemodel --img_dir /path/to/images/ --out_dir /path/to/outfeat/ --num_bbox 100,100 --feat_name pool5_flat
  ```
  - For flickr30k or referit we output the images features in directory 'data/\[flickr30k, referit\]/features/bottom-up-feats/' by default. And for refcoco/refcoco+ we output the images features in 'data/mscoco/features/bottom-up-feats/train2014'.

- 4. Download Annotation files, **we preprocess the annotations of flickr30k-entities, referit, refcoco, refcoco+** which makes all kind of data to be in same format, download our processed **annotations [here, BaiduYun](https://pan.baidu.com/s/1Qd2O9Zp5OzaGqPhEENCA2A), then unzip these zip files in directory './data'**. We will release the code for preprocessing annotation in directory './preprocess'.

- 5. Modify the paths in the config file to adapt to your own environment, set data loader threads and images features dir and images dir in yaml config files in directory './config/experiments/'.


## Training
  - flickr30k-entities
    ```
    python train_net.py --gpu_id 0 --train_split train --val_split val --cfg config/experiments/flickr30k-kld-bbox_reg.yaml
    ```
  - referit
    ```
    python train_net.py --gpu_id 0 --train_split train --val_split val --cfg config/experiments/referit-kld-bbox_reg.yaml
    ```
  - refcoco
    ```
    python train_net.py --gpu_id 0 --train_split train --val_split val --cfg config/experiments/refcoco-kld-bbox_reg.yaml
    ```
  - refcoco+
    ```
    python train_net.py --gpu_id 0 --train_split train --val_split val --cfg config/experiments/refcoco+-kld-bbox_reg.yaml
    ```
- Output model will be put in directory './models'
- Validation log output will be writen in directory './log'

## Testing
  - flickr30k-entities
    ```
    python test_net.py --gpu_id 0 --test_split test --batchsize 64 --test_net pretrained_model/flickr30k/test.prototxt --pretrained_model pretrained_model/flickr30k/final.caffemodel --cfg config/experiments/flickr30k-kld-bbox_reg.yaml
    ```
  - referit
    ```
    python test_net.py --gpu_id 0 --test_split test --batchsize 64 --test_net pretrained_model/referit/test.prototxt --pretrained_model pretrained_model/referit/final.caffemodel --cfg config/experiments/referit-kld-bbox_reg.yaml
    ```
  - refcoco
    ```
    python test_net.py --gpu_id 0 --test_split test --batchsize 64 --test_net pretrained_model/refcoco/test.prototxt --pretrained_model pretrained_model/refcoco/final.caffemodel --cfg config/experiments/refcoco-kld-bbox_reg.yaml
    ```
  - refcoco+
    ```
    python test_net.py --gpu_id 0 --test_split test --batchsize 64 --test_net pretrained_model/refcoco+/test.prototxt --pretrained_model pretrained_model/refcoco+/final.caffemodel --cfg config/experiments/refcoco+-kld-bbox_reg.yaml
    ```


## Citation
If the codes are helpful for your research, please cite
```
@article{yu2018rethining,
  title={Rethinking Diversified and Discriminative Proposal Generation for Visual Grounding},
  author={Yu, Zhou and Yu, Jun and Xiang, Chenchao, Zhao, Zhou, Tian, Qi and Tao, Dacheng},
  journal={International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2018}
}
```

