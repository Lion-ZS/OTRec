# OTRec: Cross-Modal Learning for Multimodal Recommendation via Optimal Transport


## Enviroment Requirement
- python 3.8
- Pytorch 1.12

## Dataset

We provide three processed datasets: Baby, Sports and Clothing.

Download from Google Drive: [Baby/Sports/Clothing](https://drive.google.com/file/d/1tpP-IQtUubSlVvYpkA61bffPKkhvT62T/view?usp=drive_link)

## Training
  ```
  cd ./src
  python main.py --dataset baby --gpu 1 
  python main.py --dataset clothing --gpu 2 
  python main.py --dataset sports --gpu 3 
  ```


If you have any question, please contact caozongsheng@iie.ac.cn.


## Acknowledgement
The structure of this code is  based on [MMRec](https://github.com/enoche/MMRec). Thank for their work.
