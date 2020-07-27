## Project for comp4651 (Stock movement prediction)
It mainly consists of 3 parts
1. Clean news
```shell
$ python clean_news.py
```
The cleaned news is stored in `./db/AAPL_clean_news_20170601_20180701.p`.

2. Train deep learning model
```shell
$ python dl2_cnn_read_pickle.py
```
It is trained by 1 year news and the best accuracy is about 63%. The model is stored in `./model/cnn_best_model.h5`.

3. Prediction
```shell
$ python prediction.py
```
Predict the stock movement by news and by day

Plot some graph

Solarized dark             |  Solarized Ocean
:-------------------------:|:-------------------------:
![1](https://github.com/samsonchan666/comp4651-Project/blob/master/report/val_acc.png)|![2](https://github.com/samsonchan666/comp4651-Project/blob/master/report/val_loss.png)

