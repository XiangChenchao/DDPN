# visualgrounding

This project is the implementation of the paper **Rethnking Diversified and Discriminative Proposal Generation for Visual Grounding**.

## Requirements
- Python version 2.7
- Pytorch 0.3 (optional, required for multi-threads)

## Pretrained Models

We release the Pretrained model in the paper. 

|   Datasets    | Flickr30k | Referit | Refcoco  | Refcoco+ | Refcoco+ |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| val   | xx%[BaiduYun]   |xx% [BaiduYun]()   | xx% [BaiduYun]()| xx% | xx% |
| test  | xx%[BaiduYun]   |xx% [BaiduYun]()   | xx% [BaiduYun]()| xx% | xx% |
| testA  | xx%[BaiduYun]   |xx% [BaiduYun]()   | xx% [BaiduYun]()| xx% | xx% |
| testB  | xx%[BaiduYun]   |xx% [BaiduYun]()   | xx% [BaiduYun]()| xx% | xx% |

## Preprocess
1. Download Images, **Images only**
  - for **flickr30k** 
    - download the [Flickr30k images](https://drive.google.com/file/d/0B_PL6p-5reUAZEM4MmRQQ2VVSlk/view?usp=sharing)
    - put flickr30k images in directory './data/flickr30k/flickr30k-images/'.
  - for **referit** download the [Referit Images](https://github.com/ronghanghu/natural-language-object-retrieval/blob/master/datasets/download_referit_dataset.sh).
    - wget -O ./datasets/ReferIt/ImageCLEF/referitdata.tar.gz http://www.eecs.berkeley.edu/~ronghang/projects/cvpr16_text_obj_retrieval/referitdata.tar.gz
    - tar -xzvf ./datasets/ReferIt/ImageCLEF/referitdata.tar.gz -C ./data/referit/ImageCLEF/
  - for **refcoco/refcoco+** download the *mscoco 2014 Images*
    - mscoco [train2014](http://images.cocodataset.org/zips/train2014.zip).
    - put mscoco train2014/val2014/test2015 image directory in directory './data/mscoco/image2014/'

2. Extract image features, we use [**bottom-up-attention**](https://github.com/peteanderson80/bottom-up-attention) as our image feature extractor. For a 3x800x800 image, we generate a 100x2048 tensor as input feature.

3. Download Annotation, we preprocess the annotations of flickr30k, referit, refcoco which makes all kind of data to be in same format, download our processed annotations [here, BaiduYun](), **then unzip these zip files in directory './data'**. We release the code for preprocessing annotation in directory './preprocess'.

## Training
1. Training Scratch
  - flickr30k
    - python train_net.py --gpu_id 0 --train_split train --val_split test --cfg config/experiments/flickr30k-kld-bbox_reg.yaml
  - referit
    - python train_net.py --gpu_id 0 --train_split train --val_split val --cfg config/experiments/referit-kld-bbox_reg.yaml
  - refcoco
    - python train_net.py --gpu_id 0 --train_split train --val_split test --cfg config/experiments/refcoco-kld-bbox_reg.yaml
  - refcoco+
    - python train_net.py --gpu_id 0 --train_split train --val_split test --cfg config/experiments/refcoco+-kld-bbox_reg.yaml
2. Output model will be put in directory './models'
3. Validation log output will be writen in directory './log'

## Testing
1. Testing Scratch
  - flickr30k
    - python test_net.py --gpu_id 0 --test_split test --batchsize 64 --cfg config/experiments/flickr30k-kld-bbox_reg.yaml
  - referit
    - python test_net.py --gpu_id 0 --test_split test --batchsize 64 --cfg config/experiments/referit-kld-bbox_reg.yaml
  - refcoco
    - python test_net.py --gpu_id 0 --test_split test --batchsize 64 --cfg config/experiments/refcoco-kld-bbox_reg.yaml
  - refcoco+
    - python test_net.py --gpu_id 0 --test_split test --batchsize 64 --cfg config/experiments/refcoco+-kld-bbox_reg.yaml
