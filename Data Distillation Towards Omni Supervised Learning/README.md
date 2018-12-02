# Data Distillation: Towards Omni-Supervised Learning（CVPR2018）  
論文URL：http://openaccess.thecvf.com/content_cvpr_2018/papers/Radosavovic_Data_Distillation_Towards_CVPR_2018_paper.pdf

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [何をしているのか？](#%E4%BD%95%E3%82%92%E3%81%97%E3%81%A6%E3%81%84%E3%82%8B%E3%81%AE%E3%81%8B)
- [どのような点を工夫したのか？](#%E3%81%A9%E3%81%AE%E3%82%88%E3%81%86%E3%81%AA%E7%82%B9%E3%82%92%E5%B7%A5%E5%A4%AB%E3%81%97%E3%81%9F%E3%81%AE%E3%81%8B)
- [Data Distillation for Keypoint Detection（提案手法）](#data-distillation-for-keypoint-detection%E6%8F%90%E6%A1%88%E6%89%8B%E6%B3%95)
  - [Keypoint Detection](#keypoint-detection)
  - [Mask-RCNN](#mask-rcnn)
  - [Data transformation](#data-transformation)
  - [Ensembling](#ensembling)
  - [Selecting predictions](#selecting-predictions)
  - [Generating keypoint annotations](#generating-keypoint-annotations)
  - [Retraining](#retraining)
- [Experiments on Keypoint Detection（実験1）](#experiments-on-keypoint-detection%E5%AE%9F%E9%A8%931)
  - [Data Splits（実験設定）](#data-splits%E5%AE%9F%E9%A8%93%E8%A8%AD%E5%AE%9A)
    - [COCO labeled images](#coco-labeled-images)
    - [COCO unlabeled images](#coco-unlabeled-images)
    - [Sports-1M static frames](#sports-1m-static-frames)
  - [Main Results（実験結果）](#main-results%E5%AE%9F%E9%A8%93%E7%B5%90%E6%9E%9C)
    - [Small-scale data as a sanity check](#small-scale-data-as-a-sanity-check)
    - [Large-scale data with similar distribution](#large-scale-data-with-similar-distribution)
    - [Large-scale data with dissimilar distribution](#large-scale-data-with-dissimilar-distribution)
    - [Small-scale data](#small-scale-data)
    - [Large-scale data，similar-distribution data](#large-scale-datasimilar-distribution-data)
    - [Large-scale data，dissimilar-distribution data](#large-scale-datadissimilar-distribution-data)
  - [Ablation Experiments（実験1の追加実験）](#ablation-experiments%E5%AE%9F%E9%A8%931%E3%81%AE%E8%BF%BD%E5%8A%A0%E5%AE%9F%E9%A8%93)
    - [Number of iterations](#number-of-iterations)
    - [Amount of unlabeled data](#amount-of-unlabeled-data)
    - [Impact of teacher quailty](#impact-of-teacher-quailty)
    - [Test-time augmentations](#test-time-augmentations)
- [Experiments on Object Detection（実験2）](#experiments-on-object-detection%E5%AE%9F%E9%A8%932)
  - [Implementation](#implementation)
  - [Object Detection Results](#object-detection-results)
    - [Small-scale data](#small-scale-data-1)
    - [Large-scale data](#large-scale-data)
- [まとめ](#%E3%81%BE%E3%81%A8%E3%82%81)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 何をしているのか？  
【問題点】大量のデータセットに対してアノテーション（ラベル付け）を行うには膨大な時間が必要であり，もっと楽に行える方法を模索したい  
【解決策】データセットを2分割し，一方のデータセットにアノテーションを行い，アノテーション済みのデータセットを用いてCNNの学習を行う．こうして得られた学習済みCNNを用いてもう一方のデータセットにアノテーションを行う  

【提案手法】Omni-Supervised Learning  
手法の分類としては半教師あり学習となる．ラベル付けされたデータセットを用いてラベル無しデータセットにアノテーションを行うことで低コストで膨大なデータセットの作成を可能にし，かつ精度の向上を図るという手法．
この提案手法を発展させると，少量のラベル付きデータセットを元にインターネット上に存在する大量のラベル無しデータセットに自動アノテーションを行うことで， 低コストで大規模なデータセットの作成が可能になる．   
  
大まかな流れは以下のようになる．  
1．データセットdatsetを2つに分割し，それぞれdataset1，dataset2とする．そして，dataset1のみアノテーションを行う  
![dataset_group](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/dataset_group.png)  
  
2．アノテーションを行ったdataset1を用いてCNNの学習を行う  
![dataest_label_train](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/dataset_label_test.png)  
  
3．学習済みのCNNを用いて，dataset2にアノテーションを行う  
![dataset_label_test](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/dataset_label_train.png)  

## どのような点を工夫したのか？  
前述した流れのうち，手順3で単純に1つのCNNのみを用いて学習しただけでは手順4で正確なアノテーションが行えない．
そこで，既存研究としてHintonらが提案したModel Distillationという手法が存在する．
このModel Distillationは日本語ではよく蒸留と呼ばれている手法である．
この手法の詳細は後程掲載する「Distilling the Konwledge in a Neural Network」の解説記事で解説を行う．
概要を記すと，teacher modelと呼ばれる大規模なネットワークの出力（hard targets）と出力の前のSoftmax関数の出力（soft targets）を用いて，student modelと呼ばれる小規模なネットワークを学習することで，大規模なteacher modelと同程度の精度の小規模なstudent modelを求めるという手法である．
より簡潔にまとめると，大規模なネットワークで得られた知識を小規模なネットワークに継承するという手法ある．
近年の研究では，様々な工夫により，teacher modelの精度を越えるstudent modelを導出することも可能であることが分かっている．
本研究の例では，大規模なネットワークが下図のようにmodel A，model B，model Cの3つのモデルからなるensemble modelである場合，このensemble modelがデータセットから得た知識を小規模なstudent modelに継承する．  
![figure3-1](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/model_distilation.png)  
  
それに対して，今回の論文では，Data Distillationと呼ばれる手法を提案している．
Model Distillationのteacher modeは単純なensemble modelであったが，提案手法ではteacher modelに工夫を施している．  
![figure3-2](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/data_distillation.png)  
提案手法の流れは以下のようになる．  
1．データセットを前述した流れのように，2つのアノテーションを行うデータセットdataset1と行わないデータセットdataset2に分割する  
2．dataset1の入力データに対してA，B，Cの3つの手法を用いてdata transformationを行う  
3．得られたデータセットをdata augumentationの手法ごとにtransform A，transform B，transform Cに分割する  
4．それぞれのデータセットに対して，対応するモデルを用いて学習を行う  
5．それぞれのモデルで得られた出力結果をensembleし，これを出力結果とする  
6．こうして得られた学習済みモデルを用いてdataset2にアノテーションを行う  
7．手動でアノテーションを行ったdataset1と自動でアノテーションを行ったdataset2を結合し，new datasetを作成する  
8．学習済みモデルをteacher modelとして，new datasetを用いてstudent modelを導出する  
また，手順5では一例として下図のように得られた各モデルの出力結果をensembleし，最終的出力結果を導出する．  
![ensemble_keypoionts](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/ensemble_keypoints.png)  

## Data Distillation for Keypoint Detection（提案手法）  
### Keypoint Detection
下図のように画像に写っている人の大まかな骨格を推定するタスク．  
![tennis_player](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/tennis_player.png)  

### Mask-RCNN  
CNNをベースとした高速な物体検知モデルであり，Faste R-CNNやFaster R-CNNの改良版．
Mask-RCNNの解説は本筋から外れるため，概要のみに留める．
Mask-RCNNは認識対象物体の候補領域を予測するRegion Proposal Network（RPN）の部分とRPNによって求められた認識対象物体の候補領域であるRegion of Interest（RoI）をクラス分類，回帰またはkeypointの予測を行う部分から構成される．
また，今回の論文では，Mask-RCNNのCNNとして，ResNet with Feature Pyramid Network（FPN）とResNeXt with FPNを採用する．  
![Mask-RCNN](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/Mask-RCNN.png)   

### Data transformation  
今回の論文では，data transformation手法として，下記に記す2種類のgeometric transformationの手法を採用する．  
1．scaling → stepsizeを100とし，[400, 1200]pixelsにリサイズ  
2．horizontal flipping → 左右反転  
これらのdata transformationは後述の実験でResNet-50などのモデルを採用した時の精度改善に大きく寄与した．  
※備考：テスト時には，テストデータに対してこれらのdata transformationは行っていない．  

### Ensembling
Mask-RCNNのRPNの部分のみをensembleした．
各モデルからの出力（RoI）を同じサイズとすることで，単純に各モデルの出力（RoI）の平均を取り，これをensembleした結果とした．
こうして得られたensemble後の出力（RoI）から作成された認識対象物体の候補領域に関するheatmapを作成し，最も高い値をとる領域をkeypoint locationとすることで，keypoint detectionを行った．  

### Selecting predictions  
手動でアノテーションが行われたデータと自動でアノテーションが行われたデータの予測結果は等しい価値を持つとみなして学習を行った．
また，自動アノテーションが行われたデータの出力結果の修正も行っていない．  

### Generating keypoint annotations  
前項と同じく，手動でkeypoint annotationが行われたデータと自動でkeypoint annotationが行われたデータについての扱いは等しくした  

### Retraining  
前述したように，student modelの学習は手動でアノテーションが行われたデータと自動でアノテーションが行われたデータの両方を用いて行われた．
学習時の取り決めは以下のように定めた．  
1．minibatchに含まれるデータの比率は(手動でアノテーションが行われたデータ数):(自動でアノテーションが行われたデータ数)=6:4とする  
2．学習率は最初0.02とし，全イテレーション数の70%と90%に到達したときに10で割る（例：イテレーション数が100の時の学習率は，0～70まで0.02，70～90まで0.002，90～100まで0.0002）  
3．train modelとstudent modelのモデル構造は同じとする  
4．student modelの重みの初期値はtrain modelの重みの値を使うか，ImageNetなどによる事前学習を行ったときに得られた重みの値を使う  

## Experiments on Keypoint Detection（実験1）  
データセットとしてCOCO datasetを用いる．
学習の精度の指標として，適合率（適合しない物体が含まれていない割合）を考慮したAP<sub>50</sub>，AP<sub>75</sub>，AP<sub>M</sub>（medium），AP<sub>L</sub>（large）を採用する．
ここで，AP<sub>50</sub>は適合率50%以上の物体を全て解としたときの誤答率（0～1）を1で引いたものを表す．  
### Data Splits（実験設定）
以下の3つデータセットを用いて実験を行う  
#### COCO labeled images  
画像数が80000枚のco-80を訓練データセットとし，画像数が35000枚のco-35をテストデータセットとする．
これらのデータセットはラベル付きデータセットである．
また，これらを結合したデータセットを今回の論文ではco-115とする．  
#### COCO unlabeled images
COCOが提供している120000枚のラベル無しデータセットをun-120とする．
このデータセットに含まれる画像は前述したラベル付きのCOCO dataset（co-115）の画像と似たような構成である．  
#### Sports-1M static frames  
Sports-1Mとはビデオデータセットであり，これらのビデオのフレームからランダムに抽出した180000枚の画像を元に作成したデータセットをs1m-180とする．
sim-180はラベル無しデータセットである．
このデータセットに含まれる画像はCOCO datasetの画像とは全く構成が似ていない．  
![sports-1M](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/sports-1M.png)  

### Main Results（実験結果）  
以下の3つのデータセットを作成し，これらを用いて実験を行う．  
#### Small-scale data as a sanity check  
小規模なデータセット．ラベル付きデータセットとラベル無しデータセットの特性は等しい  
ラベル付きデータセット：co-35，ラベル無しデータセット：co-80  
#### Large-scale data with similar distribution  
大規模なデータセット．ラベル付きデータセットとラベル無しデータセットの特性は等しい  
ラベル付きデータセット：co-115，ラベル無しデータセット：un-120  
#### Large-scale data with dissimilar distribution  
大規模なデータセット．ラベル付きデータセットとラベル無しデータセットの特性は異なる  
ラベル付きデータセット：co-115，ラベル無しデータセット：s1m-180  
  
次に，前述した3つのデータセットのそれぞれの結果とその考察を行っていく．  

#### Small-scale data  
下の表を見ると，ラベル付きデータセット（co-35）のみを用いた場合と比較して，全ての指標においてラベル無しデータセット（co-80）も用いた場合の方が精度が上回っていることが確認できる．  
![small-scale-data](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/small-scale-data.png)  

#### Large-scale data，similar-distribution data  
下の表を見ると，ラベル付きデータセット（co-115）のみを用いた場合と比較して，全ての指標においてラベル無しデータセット（un-120）も用いた場合の方が精度が上回っていることが確認できる．
また，どのモデルを採用した場合でも同じことが言えることが分かる．  
![large-scale-similar](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/large-scale-similar.png)  

#### Large-scale data，dissimilar-distribution data  
下の表を見ると，ラベル付きデータセット（co-115）のみを用いた場合と比較して，全ての指標においてラベル無しデータセット（s1m-180）も用いた場合の方が精度が上回っていることが確認できる．
また，どのモデルを採用した場合でも同様のことが言えることが分かる．
この結果は，所有しているラベル付きデータセットとは関係の無い画像データをインターネット上などから大量に取得することでラベル無しデータセットを作成し，手持ちのラベル付きデータセットとして扱うことで精度を上げるということが将来的に可能になるかもしれないことを示唆している．  
![large-scale-dissimilar](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/large-scale-dissimilar.png)  
  
下図にこの時の学習結果の例をいくつか示す．  
![selected_results](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/selected_results.png)

### Ablation Experiments（実験1の追加実験）  
#### Number of iterations  
ここでは，ラベル無しデータセットを用いた場合と用いなかった場合のイテレーション回数とその時の学習精度の関係について論じる．  
これらの結果は下の表のようになった．
この表を見ると，ラベル無しデータセットを用いなかった場合は，イテレーション回数が130000の時の精度が最も高くなっているのに対して，用いた場合は，ほとんどの場合，イテレーション回数が3600000の時の精度が最も高くなっていることが分かる．
これは，ラベル無しデータセットを用いなかった場合は，学習の途中で精度が悪くなり，過学習を起こしているのに対して，用いた場合は，学習が進むにつれ，精度が収束していき，過学習を起こしていないことが分かる．  
![training-iterations](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/training-iterations.png)  

#### Amount of unlabeled data  
ラベル付きデータセットとラベル無しデータセットの比率を変えた場合にどのように学習精度が変化するのかを検証する．
データセットはラベル付きデータセットがco-115であり，ラベル無しデータセットがun-120である．
モデルはRes-Net50を使用．  
(ラベル付きデータセット数):(ラベル無しデータセット数)=1:1+ρとしたときに，ρを変化させたときの学習精度の推移は下のグラフのようになった．
オレンジ色の波線は(ラベル付きデータセット数):(ラベル無しデータセット数)=1:1のときの学習精度を表す．
下のグラフを見ると，ρが増加する（ラベル無しデータセットの比率が大きくなる）と学習精度が向上していることが確認できる．  
![fraction-of-unlabeled-images](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/fraction-of-unlabeled-images.png)  

#### Impact of teacher quailty
前述した大規模ネットワークであるteacher modelの精度と蒸留後にteacher modelの知識を継承した小規模なネットワークstudent modelの精度の関係について検証する．
結果は下のグラフのようになった．
このグラフから，teacher modelの精度が良いほどstudent modelの精度も良くなることが分かる．  
![impact-of-teacher-quality](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/impact-of-teacher-quality.png)  

#### Test-time augmentations
これまでの実験ではテスト時にテストデータに対してdata transformation（augmentation）を行ってこなかった．
ここでは，テスト時にも訓練時と同様のdata transformationを行うことで学習精度にどのような影響を及ぼすかを確認する．
結果は下のグラフのようになった．
このグラフから，テスト時にdata transformationを行うことで学習精度が向上することが確認された．  
![test-time-augmentation](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/test-time-augmentation.png)  

## Experiments on Object Detection（実験2）  
データセットはCOCO datsetを用いる．  
### Implementation  
今回の論文では，物体検知器としてFPN backboneであり，認識物体領域の決定に大きくかかわるROIAlignを改良したFaster R-CNNを用いる．
また，RPNの部分は前述したkeypoint detectionに関する実験時と同様にする．  
各モデルの出力として得られたbounding boxの位置の組み合わせを多数決で決定し，ensemble modelの結果として出力する．  
このようにして，un-120に自動で付与したアノテーションは下図のようになる．  
![object-detection-annotations](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/object-detection-annotations.png)  

### Object Detection Results  
実験は以下の2つのデータセットを用いて行う．  
#### Small-scale data
小規模なデータセット．ラベル付きデータセットとラベル無しデータセットの特性は等しい  
ラベル付きデータセット：co-35，ラベル無しデータセット：co-80  
結果は以下の表のようになった．
この表から，ラベル付きデータセット（co-35）のみを用いた場合と比較して，全ての指標においてラベル無しデータセット（co-80）も用いた場合の方が精度が上回っていることが確認できる．  
![small-scale-data-object](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/small-scale-data-object.png)

#### Large-scale data
大規模なデータセット．ラベル付きデータセットとラベル無しデータセットの特性は等しい  
ラベル付きデータセット：co-115，ラベル無しデータセット：un-120  
結果は以下の表のようになった．
この表から，ラベル付きデータセット（co-115）のみを用いた場合と比較して，全ての指標においてラベル無しデータセット（un-120）も用いた場合の方が精度が上回っていることが確認できる．  
![large-scale-data-object](https://github.com/kurusugawa-computer/krsml-survey/blob/master/Survey-on-Annotation-Method/Data%20Distillation-%20Towards%20Omni-Superverised%20Learning/pictures/large-scale-data-object.png)  

## まとめ  
Keypoint DetectionとObject Detectionの2種類の実験をそれぞれ行い，提案手法であるOmni-Supervised Learningの効果を確認した．
提案手法を用いることで，以下のような利点が得られることが分かった．  
1．ラベル付けされたデータセットを用いた場合と比較して，ラベル付けデータセット+ラベル無しデータセットを用いた提案手法の方が学習精度が向上する  
2．また，過学習も避けることができる  
さらに，この他にもいくつかの実験を行うことで，提案手法の利点を解明した．
提案手法の最終目的は，少量のラベル付きデータセットを用いて，インターネット上に存在する膨大な量のラベル無しデータセットに自動アノテーションを行い，低コストで大規模なデータセットの作成を可能にするというものである．
今回の研究を通して，この最終目的を達成することができる可能性があることをを示唆したと言える．  
