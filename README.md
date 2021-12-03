# BelgiumTSC-pytorch
使用Resnet基础块构建的深度学习网络，训练交通标志数据集BelgiumTSC，训练框架为pytorch=1.1,python=3.7。

1  Prepare the data set
首先准备数据集，这里给出了百度网盘链接
You can get the data set from Baidu Cloud Disk
Link: https://pan.baidu.com/s/1JYWEFYFJCSRsVPmBfkauPQ Password: wqtv

2 Clone code
复制本仓库的代码
clone https://github.com/cqfdch/BelgiumTSC-pytorch.git

3 Change the data set path in the code to your own data set path
然后将train.py代码中的数据集路径改成下载后的数据集路径
即：train_loader,validate_loader=bs_loader.get_train_valid_loader('D:\\Networkers\\cnn-ga-master\\data',bitch_size=32,num_worker=0)里面的路径换成自己的路径就行，默认CPU训练，如果使用GPU，则num_workers=8(看自己GPU大小，设置bitch_size和num_workers)


4 Run train.py
运行train.py脚本
