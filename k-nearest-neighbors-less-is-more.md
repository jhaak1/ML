<a href="https://www.kaggle.com/code/jeremyhaakenson/k-nearest-neighbors-less-is-more?scriptVersionId=285329752" target="_blank"><img align="left" alt="Kaggle" title="Open in Kaggle" src="https://kaggle.com/static/images/open-in-kaggle.svg"></a>

In this notebook, I will attempt to improve the provided data by imputing all of the 0s using k-nearest neighbors.



First, I will load the necessary packages.


```R
library(skimr)
library(dplyr)
library(DMwR2)
```

    
    Attaching package: ‘dplyr’
    
    
    The following objects are masked from ‘package:stats’:
    
        filter, lag
    
    
    The following objects are masked from ‘package:base’:
    
        intersect, setdiff, setequal, union
    
    
    Registered S3 method overwritten by 'quantmod':
      method            from
      as.zoo.data.frame zoo 
    


Next, I willl read in the sample submission file.


```R
train = read.csv('/kaggle/input/playground-series-s3e21/sample_submission.csv')
```

Then, I will look at the data using the skim() function.


```R
#skim(train)
```

There are no missing values, but there are 0s that need to be imputed.

NO2_4 has a minimum of -4, which appears to be an outlier.

Many of the features appear to be right-skewed.

Next, I will slice off the first two columns (id and target).  These can be added back later.


```R
train.work = train[3:37]
```

Now I will replace 0s with NA, so that these values can be imputed.


```R
train.work[train.work == 0] <- NA
```

Next, I will impute NAs (previously 0s) using k-nearest neighbors.


```R
knn7 = knnImputation(train.work, k = 7)

#skim(knn7)
```

The minimum for NO2_4 looks like an outlier.  I will replace this value with .1 (the next lowest value).


```R
knn7$NO2_4[3167] = .1
#skim(knn7)
```

Finally, I will add the id and target columns back, round to three decimal places, and submit.


```R
write.csv(cbind.data.frame(train[1:2], round(knn7, 3)), 'submission.csv', row.names = F)
```
