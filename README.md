# 2022_05_Kamiya_Segmentation

### 概要
このリポジトリはモデルにSegNetとU-netを，データセットにARC DatasetとCOCOを用いたセマンティックセグメンテーションを行う．
モデルとデータセットの選択をコマンドラインオプションで指定．
生成画像を作成し，mIoUとAccuracyで定量的に評価する．


### 実行環境　
requirement.txtを参照


### データセット
COCO
https://cocodataset.org/#download からダウンロード．
保存先をargs.coco_dirで指定

ARC Dataset
http://mprg.jp/publications/c093 を参照
保存先をargs._dirで指定


### モデル
SegNet
https://github.com/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook
を参照

U-net
https://github.com/milesial/Pytorch-UNet
を参照


### 実行方法
モデルをSegNet，データセットをARC Dataset，エポック数を400で実行する場合

```sh
python main.py -m SegNet -d ARC -e 400
```

モデルをU-net，データセットをCOCO，エポック数を30で実行する場合

```sh
python main.py -m Unet -d COCO -e 30
```
