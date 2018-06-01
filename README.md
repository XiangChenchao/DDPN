# visualgrounding

This project is the implementation of the paper **Rethnking Diversified and Discriminative Proposal Generation for Visual Grounding**.

## Requirements
- Python version 2.7
- Pytorch 0.3 (optional, required for multi-threads)

## Pretrained Models

We release the Pretrained model in the paper. 

|   Datasets    | Flickr30k | Referit | Refcoco  | Refcoco+ |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| val   | 72.65   | 63.63%  | 77.18%  | 65.12%  |
| test  | 73.34   | 63.50%  | 76.70%  | 63.65%  |
| testA |         |         | 80.57%  | 70.54%  |
| testB |         |         | 72.42%  | 55.59%  |

## Preprocess
1. Download Images, **Images only**
  - **flickr30k** 
    - download the [Flickr30k images](https://drive.google.com/file/d/0B_PL6p-5reUAZEM4MmRQQ2VVSlk/view?usp=sharing)
    - put flickr30k images in directory './data/flickr30k/flickr30k-images/'.
  - **referit** download the [Referit Images](https://github.com/ronghanghu/natural-language-object-retrieval/blob/master/datasets/download_referit_dataset.sh).
    - wget -O ./datasets/ReferIt/ImageCLEF/referitdata.tar.gz http://www.eecs.berkeley.edu/~ronghang/projects/cvpr16_text_obj_retrieval/referitdata.tar.gz
    - tar -xzvf ./datasets/ReferIt/ImageCLEF/referitdata.tar.gz -C ./data/referit/ImageCLEF/
  - **refcoco/refcoco+** download the *mscoco 2014 Images*
    - mscoco [train2014](http://images.cocodataset.org/zips/train2014.zip).
    - put mscoco train2014/val2014/test2015 image directory in directory './data/mscoco/image2014/'

2. Extract image features, we use [**bottom-up-attention**](https://github.com/peteanderson80/bottom-up-attention) as our image feature extractor. For a 3x800x800 image, we generate a 100x2048 tensor as input feature.

3. Download Annotation, we preprocess the annotations of flickr30k, referit, refcoco which makes all kind of data to be in same format, download our processed annotations [here, BaiduYun](https://pan.baidu.com/s/1Qd2O9Zp5OzaGqPhEENCA2A), **then unzip these zip files in directory './data'**. We release the code for preprocessing annotation in directory './preprocess'.


## Pretrained Models
1. Download models [here, BaiduYun](https://pan.baidu.com/s/1QiLu28UoOePCHe2W_2gNVA)
2. Unzip the zip models file in directory './pretrained_model'.

## Training
1. Training Scratch
  - flickr30k
    - python train_net.py --gpu_id 0 --train_split train --val_split val --cfg config/experiments/flickr30k-kld-bbox_reg.yaml
  - referit
    - python train_net.py --gpu_id 0 --train_split train --val_split val --cfg config/experiments/referit-kld-bbox_reg.yaml
  - refcoco
    - python train_net.py --gpu_id 0 --train_split train --val_split val --cfg config/experiments/refcoco-kld-bbox_reg.yaml
  - refcoco+
    - python train_net.py --gpu_id 0 --train_split train --val_split val --cfg config/experiments/refcoco+-kld-bbox_reg.yaml
2. Output model will be put in directory './models'
3. Validation log output will be writen in directory './log'

## Testing
1. Testing Scratch
  - flickr30k
    - python test_net.py --gpu_id 0 --test_split test --batchsize 64 --test_net pretrained_model/flickr30k/test.prototxt --pretrained_model pretrained_model/flickr30k/final.caffemodel --cfg config/experiments/flickr30k-kld-bbox_reg.yaml
  - referit
    - python test_net.py --gpu_id 0 --test_split test --batchsize 64 --test_net pretrained_model/referit/test.prototxt --pretrained_model pretrained_model/referit/final.caffemodel --cfg config/experiments/referit-kld-bbox_reg.yaml
  - refcoco
    - python test_net.py --gpu_id 0 --test_split test --batchsize 64 --test_net pretrained_model/refcoco/test.prototxt --pretrained_model pretrained_model/refcoco/final.caffemodel --cfg config/experiments/refcoco-kld-bbox_reg.yaml
  - refcoco+
    - python test_net.py --gpu_id 0 --test_split test --batchsize 64 --test_net pretrained_model/refcoco+/test.prototxt --pretrained_model pretrained_model/refcoco+/final.caffemodel --cfg config/experiments/refcoco+-kld-bbox_reg.yaml
