<a href="https://www.kaggle.com/code/jeremyhaakenson/generalized-additive-model-to-predict-flooding?scriptVersionId=285330664" target="_blank"><img align="left" alt="Kaggle" title="Open in Kaggle" src="https://kaggle.com/static/images/open-in-kaggle.svg"></a>

In this notebook, I will use a generalized additive model to predict flooding.

# GAM

Load the necessary packages.


```R
library(gam)
```

    Loading required package: splines
    
    Loading required package: foreach
    
    Loaded gam 1.22-3
    
    


Read in the data.


```R
train = read.csv('/kaggle/input/playground-series-s4e5/train.csv')
test = read.csv('/kaggle/input/playground-series-s4e5/test.csv')
```

Add the feature suggested by @ambrosm, which is the sum of all of the original features.


```R
train$allsum = rowSums(train[2:21])
test$allsum = rowSums(test[2:21])
```

Make the generalized additive model.


```R
gam4 = gam(FloodProbability ~ (. -id)^2, data = train)
gam4.pred = suppressWarnings(predict(gam4, test))
gam4.sub = cbind.data.frame(test[1], gam4.pred)
colnames(gam4.sub)[2] = 'FloodProbability'
write.csv(gam4.sub, 'submission.csv', row.names = F)
```

This was pretty quick and easy.  This model might not perform as well as a neural network with this data, but I think it will be an important part of my final ensemble.
