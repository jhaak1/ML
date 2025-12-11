<a href="https://www.kaggle.com/code/jeremyhaakenson/crab-xgboost?scriptVersionId=285333186" target="_blank"><img align="left" alt="Kaggle" title="Open in Kaggle" src="https://kaggle.com/static/images/open-in-kaggle.svg"></a>

#EDA

**Load packages.**


```R
library(skimr)
library(moments)
library(dplyr)
library(caret)
library(xgboost)
```

    
    Attaching package: ‘dplyr’
    
    
    The following objects are masked from ‘package:stats’:
    
        filter, lag
    
    
    The following objects are masked from ‘package:base’:
    
        intersect, setdiff, setequal, union
    
    
    Loading required package: ggplot2
    
    Loading required package: lattice
    
    
    Attaching package: ‘caret’
    
    
    The following object is masked from ‘package:httr’:
    
        progress
    
    
    
    Attaching package: ‘xgboost’
    
    
    The following object is masked from ‘package:dplyr’:
    
        slice
    
    


**Load data.**


```R
train = read.csv('/kaggle/input/playground-series-s3e16/train.csv')
test = read.csv('/kaggle/input/playground-series-s3e16/test.csv')
```

**Examine data.**


```R
any(is.na(train))
any(is.na(test))
```


FALSE



FALSE


**There are no missing values in train or test.**

**Convert Sex to categorical.**


```R
train$Sex = as.factor(train$Sex)
test$Sex = as.factor(test$Sex)
```

#Feature Engineering

**I will add 3 new features: Shell.Percent, Meat.Percent, and Viscera.Percent.**


```R
train.fe = train %>%
  mutate(Shell.Percent = Shell.Weight/Weight,
         Meat.Percent = Shucked.Weight/Weight,
         Viscera.Percent = Viscera.Weight/Weight)

test.fe = test %>%
    mutate(Shell.Percent = Shell.Weight/Weight,
          Meat.Percent = Shucked.Weight/Weight,
          Viscera.Percent = Viscera.Weight/Weight)

head(train.fe)
```


<table class="dataframe">
<caption>A data.frame: 6 × 13</caption>
<thead>
	<tr><th></th><th scope=col>id</th><th scope=col>Sex</th><th scope=col>Length</th><th scope=col>Diameter</th><th scope=col>Height</th><th scope=col>Weight</th><th scope=col>Shucked.Weight</th><th scope=col>Viscera.Weight</th><th scope=col>Shell.Weight</th><th scope=col>Age</th><th scope=col>Shell.Percent</th><th scope=col>Meat.Percent</th><th scope=col>Viscera.Percent</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0</td><td>I</td><td>1.5250</td><td>1.1750</td><td>0.3750</td><td>28.97319</td><td>12.728926</td><td> 6.647958</td><td> 8.348928</td><td> 9</td><td>0.2881605</td><td>0.4393346</td><td>0.2294521</td></tr>
	<tr><th scope=row>2</th><td>1</td><td>I</td><td>1.1000</td><td>0.8250</td><td>0.2750</td><td>10.41844</td><td> 4.521745</td><td> 2.324659</td><td> 3.401940</td><td> 8</td><td>0.3265306</td><td>0.4340136</td><td>0.2231293</td></tr>
	<tr><th scope=row>3</th><td>2</td><td>M</td><td>1.3875</td><td>1.1125</td><td>0.3750</td><td>24.77746</td><td>11.339800</td><td> 5.556502</td><td> 6.662133</td><td> 9</td><td>0.2688787</td><td>0.4576659</td><td>0.2242563</td></tr>
	<tr><th scope=row>4</th><td>3</td><td>F</td><td>1.7000</td><td>1.4125</td><td>0.5000</td><td>50.66056</td><td>20.354941</td><td>10.991839</td><td>14.996885</td><td>11</td><td>0.2960269</td><td>0.4017907</td><td>0.2169703</td></tr>
	<tr><th scope=row>5</th><td>4</td><td>I</td><td>1.2500</td><td>1.0125</td><td>0.3375</td><td>23.28911</td><td>11.977664</td><td> 4.507570</td><td> 5.953395</td><td> 8</td><td>0.2556299</td><td>0.5143031</td><td>0.1935484</td></tr>
	<tr><th scope=row>6</th><td>5</td><td>M</td><td>1.5000</td><td>1.1750</td><td>0.4125</td><td>28.84562</td><td>13.409313</td><td> 6.789705</td><td> 7.937860</td><td>10</td><td>0.2751843</td><td>0.4648649</td><td>0.2353808</td></tr>
</tbody>
</table>



#Correlation

**Look for correlation between variables.**


```R
cor(train.fe[3:13], method = 'spearman')
```


<table class="dataframe">
<caption>A matrix: 11 × 11 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>Length</th><th scope=col>Diameter</th><th scope=col>Height</th><th scope=col>Weight</th><th scope=col>Shucked.Weight</th><th scope=col>Viscera.Weight</th><th scope=col>Shell.Weight</th><th scope=col>Age</th><th scope=col>Shell.Percent</th><th scope=col>Meat.Percent</th><th scope=col>Viscera.Percent</th></tr>
</thead>
<tbody>
	<tr><th scope=row>Length</th><td> 1.00000000</td><td> 0.98394293</td><td> 0.908566127</td><td> 0.976708090</td><td> 0.96017164</td><td> 0.95815094</td><td> 0.953545164</td><td> 0.668712061</td><td>-0.24645745</td><td> 0.093872882</td><td> 0.016969047</td></tr>
	<tr><th scope=row>Diameter</th><td> 0.98394293</td><td> 1.00000000</td><td> 0.912796255</td><td> 0.976879525</td><td> 0.95622391</td><td> 0.95722974</td><td> 0.958793115</td><td> 0.678573500</td><td>-0.22870583</td><td> 0.077617472</td><td> 0.012382236</td></tr>
	<tr><th scope=row>Height</th><td> 0.90856613</td><td> 0.91279626</td><td> 1.000000000</td><td> 0.926750150</td><td> 0.89072234</td><td> 0.91075326</td><td> 0.930749648</td><td> 0.704477606</td><td>-0.14871579</td><td> 0.006959979</td><td> 0.019731057</td></tr>
	<tr><th scope=row>Weight</th><td> 0.97670809</td><td> 0.97687952</td><td> 0.926750150</td><td> 1.000000000</td><td> 0.97543504</td><td> 0.97516738</td><td> 0.971981870</td><td> 0.689562366</td><td>-0.26871179</td><td> 0.071433199</td><td>-0.003093649</td></tr>
	<tr><th scope=row>Shucked.Weight</th><td> 0.96017164</td><td> 0.95622391</td><td> 0.890722340</td><td> 0.975435037</td><td> 1.00000000</td><td> 0.95008090</td><td> 0.926190368</td><td> 0.612695724</td><td>-0.34305384</td><td> 0.254799238</td><td>-0.011182225</td></tr>
	<tr><th scope=row>Viscera.Weight</th><td> 0.95815094</td><td> 0.95722974</td><td> 0.910753257</td><td> 0.975167376</td><td> 0.95008090</td><td> 1.00000000</td><td> 0.946806919</td><td> 0.676960216</td><td>-0.26731782</td><td> 0.059579471</td><td> 0.181770610</td></tr>
	<tr><th scope=row>Shell.Weight</th><td> 0.95354516</td><td> 0.95879311</td><td> 0.930749648</td><td> 0.971981870</td><td> 0.92619037</td><td> 0.94680692</td><td> 1.000000000</td><td> 0.736481013</td><td>-0.07306613</td><td>-0.018817547</td><td>-0.008214282</td></tr>
	<tr><th scope=row>Age</th><td> 0.66871206</td><td> 0.67857350</td><td> 0.704477606</td><td> 0.689562366</td><td> 0.61269572</td><td> 0.67696022</td><td> 0.736481013</td><td> 1.000000000</td><td> 0.05318348</td><td>-0.214272506</td><td> 0.003205292</td></tr>
	<tr><th scope=row>Shell.Percent</th><td>-0.24645745</td><td>-0.22870583</td><td>-0.148715786</td><td>-0.268711794</td><td>-0.34305384</td><td>-0.26731782</td><td>-0.073066131</td><td> 0.053183481</td><td> 1.00000000</td><td>-0.357719114</td><td> 0.029330135</td></tr>
	<tr><th scope=row>Meat.Percent</th><td> 0.09387288</td><td> 0.07761747</td><td> 0.006959979</td><td> 0.071433199</td><td> 0.25479924</td><td> 0.05957947</td><td>-0.018817547</td><td>-0.214272506</td><td>-0.35771911</td><td> 1.000000000</td><td>-0.037396327</td></tr>
	<tr><th scope=row>Viscera.Percent</th><td> 0.01696905</td><td> 0.01238224</td><td> 0.019731057</td><td>-0.003093649</td><td>-0.01118223</td><td> 0.18177061</td><td>-0.008214282</td><td> 0.003205292</td><td> 0.02933013</td><td>-0.037396327</td><td> 1.000000000</td></tr>
</tbody>
</table>



**Length, Diameter, Height, Weight, Shucked.Weight, Viscera.Weight, and Shell.Weight are all highly correlated with each other.**

**I will keep Shell.Weight since it has the highest correlation with Age out of those 7 variables.**

**None of the variables in train are highly correlated with Age (rho >= .75).**


```R
train.cor = train.fe[c(1:2, 9:13)]
head(train.cor)

test.cor = test.fe[c(1:2, 9, 10:12)]
head(test.cor)
```


<table class="dataframe">
<caption>A data.frame: 6 × 7</caption>
<thead>
	<tr><th></th><th scope=col>id</th><th scope=col>Sex</th><th scope=col>Shell.Weight</th><th scope=col>Age</th><th scope=col>Shell.Percent</th><th scope=col>Meat.Percent</th><th scope=col>Viscera.Percent</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0</td><td>I</td><td> 8.348928</td><td> 9</td><td>0.2881605</td><td>0.4393346</td><td>0.2294521</td></tr>
	<tr><th scope=row>2</th><td>1</td><td>I</td><td> 3.401940</td><td> 8</td><td>0.3265306</td><td>0.4340136</td><td>0.2231293</td></tr>
	<tr><th scope=row>3</th><td>2</td><td>M</td><td> 6.662133</td><td> 9</td><td>0.2688787</td><td>0.4576659</td><td>0.2242563</td></tr>
	<tr><th scope=row>4</th><td>3</td><td>F</td><td>14.996885</td><td>11</td><td>0.2960269</td><td>0.4017907</td><td>0.2169703</td></tr>
	<tr><th scope=row>5</th><td>4</td><td>I</td><td> 5.953395</td><td> 8</td><td>0.2556299</td><td>0.5143031</td><td>0.1935484</td></tr>
	<tr><th scope=row>6</th><td>5</td><td>M</td><td> 7.937860</td><td>10</td><td>0.2751843</td><td>0.4648649</td><td>0.2353808</td></tr>
</tbody>
</table>




<table class="dataframe">
<caption>A data.frame: 6 × 6</caption>
<thead>
	<tr><th></th><th scope=col>id</th><th scope=col>Sex</th><th scope=col>Shell.Weight</th><th scope=col>Shell.Percent</th><th scope=col>Meat.Percent</th><th scope=col>Viscera.Percent</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>74051</td><td>I</td><td>2.721552</td><td>0.3157895</td><td>0.4243421</td><td>0.2006579</td></tr>
	<tr><th scope=row>2</th><td>74052</td><td>I</td><td>3.968930</td><td>0.2559415</td><td>0.4533821</td><td>0.2093236</td></tr>
	<tr><th scope=row>3</th><td>74053</td><td>F</td><td>4.819415</td><td>0.3307393</td><td>0.3813230</td><td>0.2665370</td></tr>
	<tr><th scope=row>4</th><td>74054</td><td>F</td><td>7.030676</td><td>0.2477522</td><td>0.4715285</td><td>0.2307692</td></tr>
	<tr><th scope=row>5</th><td>74055</td><td>I</td><td>3.331066</td><td>0.2831325</td><td>0.4698795</td><td>0.2096386</td></tr>
	<tr><th scope=row>6</th><td>74056</td><td>M</td><td>8.079607</td><td>0.3253425</td><td>0.3515982</td><td>0.2300228</td></tr>
</tbody>
</table>



#Skew

**Look for skew one variable at a time.**

**Shell.Weight**


```R
skewness(train.cor$Shell.Weight)
```


0.277453349401448


**Shell.Percent**


```R
skewness(train.cor$Shell.Percent)
skewness(sqrt(train.cor$Shell.Percent))
```


3.96887217836599



1.09771735525932



```R
train.cor$Shell.Percent = sqrt(train.cor$Shell.Percent)
test.cor$Shell.Percent = sqrt(test.cor$Shell.Percent)
```

**Meat.Percent**


```R
skewness(train.cor$Meat.Percent)
skewness(sqrt(train.cor$Meat.Percent))
skewness((train.cor$Meat.Percent)^(1/3))
skewness(log(train.cor$Meat.Percent))
```


28.1392651433865



6.74636190401158



3.70609835924653



0.786980584068327



```R
train.cor$Meat.Percent = log(train.cor$Meat.Percent)
test.cor$Meat.Percent = log(test.cor$Meat.Percent)
```

**Viscera.Percent**


```R
skewness(train.cor$Viscera.Percent)
```


1.72301437637722


#Outliers

**Define a function for replacing outliers.**


```R
outlier_norm <- function(x){
  qntile <- quantile(x, probs=c(.25, .75))
  H <- 1.5 * IQR(x)
  x[x < median(x) - H] <- median(x) - H
  x[x > median(x) + H] <- median(x) + H
  return(x)
}
```

**Look for outliers.**


```R
head(train.cor)
```


<table class="dataframe">
<caption>A data.frame: 6 × 7</caption>
<thead>
	<tr><th></th><th scope=col>id</th><th scope=col>Sex</th><th scope=col>Shell.Weight</th><th scope=col>Age</th><th scope=col>Shell.Percent</th><th scope=col>Meat.Percent</th><th scope=col>Viscera.Percent</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0</td><td>I</td><td> 8.348928</td><td> 9</td><td>0.5368058</td><td>-0.8224939</td><td>0.2294521</td></tr>
	<tr><th scope=row>2</th><td>1</td><td>I</td><td> 3.401940</td><td> 8</td><td>0.5714286</td><td>-0.8346794</td><td>0.2231293</td></tr>
	<tr><th scope=row>3</th><td>2</td><td>M</td><td> 6.662133</td><td> 9</td><td>0.5185352</td><td>-0.7816158</td><td>0.2242563</td></tr>
	<tr><th scope=row>4</th><td>3</td><td>F</td><td>14.996885</td><td>11</td><td>0.5440835</td><td>-0.9118239</td><td>0.2169703</td></tr>
	<tr><th scope=row>5</th><td>4</td><td>I</td><td> 5.953395</td><td> 8</td><td>0.5055986</td><td>-0.6649425</td><td>0.1935484</td></tr>
	<tr><th scope=row>6</th><td>5</td><td>M</td><td> 7.937860</td><td>10</td><td>0.5245801</td><td>-0.7660085</td><td>0.2353808</td></tr>
</tbody>
</table>




```R
boxplot(train.cor$Shell.Weight)
```


    
![png](crab-xgboost_files/crab-xgboost_29_0.png)
    



```R
train.out = train.cor %>%
    mutate(Shell.Weight = outlier_norm(Shell.Weight))

test.out = test.cor %>%
    mutate(Shell.Weight = outlier_norm(Shell.Weight))

boxplot(train.out$Shell.Weight)
```


    
![png](crab-xgboost_files/crab-xgboost_30_0.png)
    



```R
boxplot(train.out$Shell.Percent)
```


    
![png](crab-xgboost_files/crab-xgboost_31_0.png)
    



```R
train.out$Shell.Percent = outlier_norm(train.out$Shell.Percent)

test.out$Shell.Percent = outlier_norm(test.out$Shell.Percent)

boxplot(train.out$Shell.Percent)
```


    
![png](crab-xgboost_files/crab-xgboost_32_0.png)
    



```R
boxplot(train.out$Meat.Percent)
```


    
![png](crab-xgboost_files/crab-xgboost_33_0.png)
    



```R
train.out$Meat.Percent = outlier_norm(train.out$Meat.Percent)

test.out$Meat.Percent = outlier_norm(test.out$Meat.Percent)

boxplot(train.out$Meat.Percent)
```


    
![png](crab-xgboost_files/crab-xgboost_34_0.png)
    



```R
boxplot(train.out$Viscera.Percent)
```


    
![png](crab-xgboost_files/crab-xgboost_35_0.png)
    



```R
train.out$Viscera.Percent = outlier_norm(train.out$Viscera.Percent)

test.out$Viscera.Percent = outlier_norm(test.out$Viscera.Percent)

boxplot(train.out$Viscera.Percent)
```


    
![png](crab-xgboost_files/crab-xgboost_36_0.png)
    


#Scaling


```R
head(train.out)
head(test.out)
```


<table class="dataframe">
<caption>A data.frame: 6 × 7</caption>
<thead>
	<tr><th></th><th scope=col>id</th><th scope=col>Sex</th><th scope=col>Shell.Weight</th><th scope=col>Age</th><th scope=col>Shell.Percent</th><th scope=col>Meat.Percent</th><th scope=col>Viscera.Percent</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0</td><td>I</td><td> 8.348928</td><td> 9</td><td>0.5368058</td><td>-0.8224939</td><td>0.2294521</td></tr>
	<tr><th scope=row>2</th><td>1</td><td>I</td><td> 3.401940</td><td> 8</td><td>0.5714286</td><td>-0.8346794</td><td>0.2231293</td></tr>
	<tr><th scope=row>3</th><td>2</td><td>M</td><td> 6.662133</td><td> 9</td><td>0.5185352</td><td>-0.7816158</td><td>0.2242563</td></tr>
	<tr><th scope=row>4</th><td>3</td><td>F</td><td>14.585818</td><td>11</td><td>0.5440835</td><td>-0.9118239</td><td>0.2169703</td></tr>
	<tr><th scope=row>5</th><td>4</td><td>I</td><td> 5.953395</td><td> 8</td><td>0.5055986</td><td>-0.6649425</td><td>0.1935484</td></tr>
	<tr><th scope=row>6</th><td>5</td><td>M</td><td> 7.937860</td><td>10</td><td>0.5245801</td><td>-0.7660085</td><td>0.2353808</td></tr>
</tbody>
</table>




<table class="dataframe">
<caption>A data.frame: 6 × 6</caption>
<thead>
	<tr><th></th><th scope=col>id</th><th scope=col>Sex</th><th scope=col>Shell.Weight</th><th scope=col>Shell.Percent</th><th scope=col>Meat.Percent</th><th scope=col>Viscera.Percent</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>74051</td><td>I</td><td>2.721552</td><td>0.5619515</td><td>-0.8572153</td><td>0.2006579</td></tr>
	<tr><th scope=row>2</th><td>74052</td><td>I</td><td>3.968930</td><td>0.5059066</td><td>-0.7910201</td><td>0.2093236</td></tr>
	<tr><th scope=row>3</th><td>74053</td><td>F</td><td>4.819415</td><td>0.5750994</td><td>-0.9641086</td><td>0.2665370</td></tr>
	<tr><th scope=row>4</th><td>74054</td><td>F</td><td>7.030676</td><td>0.4977472</td><td>-0.7517758</td><td>0.2307692</td></tr>
	<tr><th scope=row>5</th><td>74055</td><td>I</td><td>3.331066</td><td>0.5321020</td><td>-0.7552790</td><td>0.2096386</td></tr>
	<tr><th scope=row>6</th><td>74056</td><td>M</td><td>8.079607</td><td>0.5703880</td><td>-1.0452663</td><td>0.2300228</td></tr>
</tbody>
</table>




```R
train.sca = scale(train.out[c(3, 5:7)])
train.scale = cbind.data.frame(train.out[1:2], train.sca, train.out[4])

test.sca = scale(test.out[3:6])
test.scale = cbind.data.frame(test[1:2], test.sca)
```

#One-hot Encoding


```R
hot1 = dummyVars('~ .', data = train.scale)
train.hot = data.frame(predict(hot1, newdata = train.scale))

hot2 = dummyVars('~ .', data = test.scale)
test.hot = data.frame(predict(hot2, newdata = test.scale))

head(train.hot)
head(test.hot)
```


<table class="dataframe">
<caption>A data.frame: 6 × 9</caption>
<thead>
	<tr><th></th><th scope=col>id</th><th scope=col>Sex.F</th><th scope=col>Sex.I</th><th scope=col>Sex.M</th><th scope=col>Shell.Weight</th><th scope=col>Shell.Percent</th><th scope=col>Meat.Percent</th><th scope=col>Viscera.Percent</th><th scope=col>Age</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0</td><td>0</td><td>1</td><td>0</td><td> 0.473452366</td><td>-0.07882564</td><td> 0.2210556</td><td> 0.51102978</td><td> 9</td></tr>
	<tr><th scope=row>2</th><td>1</td><td>0</td><td>1</td><td>0</td><td>-0.940597781</td><td> 1.02610615</td><td> 0.1097798</td><td> 0.26353102</td><td> 8</td></tr>
	<tr><th scope=row>3</th><td>2</td><td>0</td><td>0</td><td>1</td><td>-0.008702269</td><td>-0.66190456</td><td> 0.5943460</td><td> 0.30764774</td><td> 9</td></tr>
	<tr><th scope=row>4</th><td>3</td><td>1</td><td>0</td><td>0</td><td> 2.256208997</td><td> 0.15343092</td><td>-0.5946891</td><td> 0.02244785</td><td>11</td></tr>
	<tr><th scope=row>5</th><td>4</td><td>0</td><td>1</td><td>0</td><td>-0.211288250</td><td>-1.07475493</td><td> 1.6597841</td><td>-0.89437749</td><td> 8</td></tr>
	<tr><th scope=row>6</th><td>5</td><td>0</td><td>0</td><td>1</td><td> 0.355952497</td><td>-0.46899009</td><td> 0.7368688</td><td> 0.74310500</td><td>10</td></tr>
</tbody>
</table>




<table class="dataframe">
<caption>A data.frame: 6 × 8</caption>
<thead>
	<tr><th></th><th scope=col>id</th><th scope=col>Sex.F</th><th scope=col>Sex.I</th><th scope=col>Sex.M</th><th scope=col>Shell.Weight</th><th scope=col>Shell.Percent</th><th scope=col>Meat.Percent</th><th scope=col>Viscera.Percent</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>74051</td><td>0</td><td>1</td><td>0</td><td>-1.14357175</td><td> 0.7227927</td><td>-0.08867063</td><td>-0.6113890</td></tr>
	<tr><th scope=row>2</th><td>74052</td><td>0</td><td>1</td><td>0</td><td>-0.78650187</td><td>-1.0622529</td><td> 0.51454487</td><td>-0.2709705</td></tr>
	<tr><th scope=row>3</th><td>74053</td><td>1</td><td>0</td><td>0</td><td>-0.54304513</td><td> 1.1415570</td><td>-1.06275426</td><td> 1.9765707</td></tr>
	<tr><th scope=row>4</th><td>74054</td><td>1</td><td>0</td><td>0</td><td> 0.08994238</td><td>-1.3221334</td><td> 0.87216495</td><td> 0.5714894</td></tr>
	<tr><th scope=row>5</th><td>74055</td><td>0</td><td>1</td><td>0</td><td>-0.96909443</td><td>-0.2279224</td><td> 0.84024173</td><td>-0.2585973</td></tr>
	<tr><th scope=row>6</th><td>74056</td><td>0</td><td>0</td><td>1</td><td> 0.39020569</td><td> 0.9914980</td><td>-1.80231776</td><td> 0.5421682</td></tr>
</tbody>
</table>



#XGBoost Model

**Put data in XGBoost format.**


```R
train.xgb <- xgb.DMatrix(label = train.hot$Age, data = as.matrix(train.hot[2:8]))
test.xgb = as.matrix(test.hot[2:8])
```

**Determine number of rounds to use after optimizing parameters (not shown).**


```R
set.seed(155)
xgb3 = xgb.cv(params = list(eta = .3, 
                            gamma = 0,
                            subsample = .8,
                            colsample_bytree = .9, 
                            reg_alpha = .08,
                            reg_lambda = 1),
              max_depth = 3,
              min_child_weight = 16,
              data = train.xgb, 
              nrounds = 400,
              objective = 'reg:absoluteerror',
              nfold = 10, 
              early_stopping_rounds = 16)
```

    [1]	train-mae:2.227586+0.010460	test-mae:2.227674+0.028890 
    Multiple eval metrics are present. Will use test_mae for early stopping.
    Will train until test_mae hasn't improved in 16 rounds.
    
    [2]	train-mae:2.139067+0.008646	test-mae:2.139432+0.027863 
    [3]	train-mae:2.052670+0.014116	test-mae:2.053258+0.028952 
    [4]	train-mae:1.965802+0.013451	test-mae:1.966430+0.024742 
    [5]	train-mae:1.894725+0.014920	test-mae:1.895596+0.024869 
    [6]	train-mae:1.825864+0.014472	test-mae:1.826756+0.028492 
    [7]	train-mae:1.761457+0.016662	test-mae:1.762185+0.032443 
    [8]	train-mae:1.704003+0.019350	test-mae:1.704856+0.036281 
    [9]	train-mae:1.656873+0.018296	test-mae:1.658232+0.034101 
    [10]	train-mae:1.614026+0.015483	test-mae:1.615800+0.031623 
    [11]	train-mae:1.575337+0.014965	test-mae:1.577368+0.032466 
    [12]	train-mae:1.546177+0.013044	test-mae:1.548481+0.031638 
    [13]	train-mae:1.520246+0.010164	test-mae:1.522775+0.027854 
    [14]	train-mae:1.501707+0.007931	test-mae:1.504420+0.025685 
    [15]	train-mae:1.486678+0.008729	test-mae:1.489599+0.024113 
    [16]	train-mae:1.473292+0.008297	test-mae:1.476423+0.022233 
    [17]	train-mae:1.461633+0.006841	test-mae:1.465000+0.021118 
    [18]	train-mae:1.451306+0.005928	test-mae:1.455005+0.021349 
    [19]	train-mae:1.442772+0.006187	test-mae:1.446551+0.020416 
    [20]	train-mae:1.436029+0.006551	test-mae:1.440012+0.019658 
    [21]	train-mae:1.430720+0.005580	test-mae:1.434769+0.018976 
    [22]	train-mae:1.426207+0.004665	test-mae:1.430528+0.018755 
    [23]	train-mae:1.422339+0.004730	test-mae:1.426816+0.017876 
    [24]	train-mae:1.418850+0.004859	test-mae:1.423557+0.017043 
    [25]	train-mae:1.415348+0.004523	test-mae:1.420031+0.016957 
    [26]	train-mae:1.411849+0.004223	test-mae:1.416614+0.016428 
    [27]	train-mae:1.409086+0.003827	test-mae:1.413977+0.016341 
    [28]	train-mae:1.406877+0.003698	test-mae:1.411907+0.016241 
    [29]	train-mae:1.404227+0.003375	test-mae:1.409396+0.015821 
    [30]	train-mae:1.402198+0.003119	test-mae:1.407472+0.015363 
    [31]	train-mae:1.400248+0.003040	test-mae:1.405727+0.014845 
    [32]	train-mae:1.398393+0.003005	test-mae:1.403984+0.014857 
    [33]	train-mae:1.396684+0.002702	test-mae:1.402433+0.014707 
    [34]	train-mae:1.395129+0.002468	test-mae:1.400893+0.014587 
    [35]	train-mae:1.393730+0.002317	test-mae:1.399683+0.014297 
    [36]	train-mae:1.392262+0.002245	test-mae:1.398337+0.014108 
    [37]	train-mae:1.391026+0.002137	test-mae:1.397253+0.014006 
    [38]	train-mae:1.389973+0.001972	test-mae:1.396235+0.013912 
    [39]	train-mae:1.388889+0.001919	test-mae:1.395268+0.013938 
    [40]	train-mae:1.387841+0.001885	test-mae:1.394242+0.013704 
    [41]	train-mae:1.386922+0.001916	test-mae:1.393362+0.013458 
    [42]	train-mae:1.386098+0.001995	test-mae:1.392673+0.013449 
    [43]	train-mae:1.385017+0.001990	test-mae:1.391741+0.013151 
    [44]	train-mae:1.384170+0.001870	test-mae:1.390971+0.013034 
    [45]	train-mae:1.383426+0.001802	test-mae:1.390306+0.012931 
    [46]	train-mae:1.382664+0.001783	test-mae:1.389662+0.012904 
    [47]	train-mae:1.381931+0.001762	test-mae:1.389007+0.012872 
    [48]	train-mae:1.381354+0.001783	test-mae:1.388426+0.012791 
    [49]	train-mae:1.380702+0.001911	test-mae:1.387877+0.012573 
    [50]	train-mae:1.380281+0.001958	test-mae:1.387528+0.012399 
    [51]	train-mae:1.379722+0.002038	test-mae:1.386944+0.012213 
    [52]	train-mae:1.379203+0.001980	test-mae:1.386637+0.012203 
    [53]	train-mae:1.378720+0.001890	test-mae:1.386146+0.011988 
    [54]	train-mae:1.378210+0.001867	test-mae:1.385726+0.012014 
    [55]	train-mae:1.377689+0.001822	test-mae:1.385263+0.011844 
    [56]	train-mae:1.377208+0.001760	test-mae:1.384809+0.011808 
    [57]	train-mae:1.376724+0.001715	test-mae:1.384373+0.011813 
    [58]	train-mae:1.376171+0.001554	test-mae:1.383948+0.011839 
    [59]	train-mae:1.375796+0.001543	test-mae:1.383751+0.011900 
    [60]	train-mae:1.375414+0.001498	test-mae:1.383463+0.011963 
    [61]	train-mae:1.375064+0.001475	test-mae:1.383173+0.011916 
    [62]	train-mae:1.374754+0.001458	test-mae:1.382920+0.011945 
    [63]	train-mae:1.374413+0.001386	test-mae:1.382622+0.011827 
    [64]	train-mae:1.374082+0.001439	test-mae:1.382348+0.011823 
    [65]	train-mae:1.373725+0.001382	test-mae:1.382003+0.011870 
    [66]	train-mae:1.373400+0.001357	test-mae:1.381756+0.011892 
    [67]	train-mae:1.372984+0.001359	test-mae:1.381356+0.011776 
    [68]	train-mae:1.372754+0.001364	test-mae:1.381232+0.011832 
    [69]	train-mae:1.372486+0.001411	test-mae:1.380994+0.011605 
    [70]	train-mae:1.372155+0.001350	test-mae:1.380737+0.011651 
    [71]	train-mae:1.371843+0.001366	test-mae:1.380512+0.011541 
    [72]	train-mae:1.371679+0.001366	test-mae:1.380446+0.011451 
    [73]	train-mae:1.371414+0.001251	test-mae:1.380333+0.011485 
    [74]	train-mae:1.371275+0.001251	test-mae:1.380153+0.011533 
    [75]	train-mae:1.371033+0.001162	test-mae:1.379984+0.011541 
    [76]	train-mae:1.370855+0.001140	test-mae:1.379881+0.011531 
    [77]	train-mae:1.370653+0.001173	test-mae:1.379771+0.011487 
    [78]	train-mae:1.370487+0.001171	test-mae:1.379768+0.011519 
    [79]	train-mae:1.370352+0.001177	test-mae:1.379610+0.011570 
    [80]	train-mae:1.370231+0.001163	test-mae:1.379603+0.011569 
    [81]	train-mae:1.370128+0.001156	test-mae:1.379606+0.011580 
    [82]	train-mae:1.369956+0.001167	test-mae:1.379451+0.011607 
    [83]	train-mae:1.369833+0.001122	test-mae:1.379435+0.011657 
    [84]	train-mae:1.369590+0.001144	test-mae:1.379277+0.011480 
    [85]	train-mae:1.369411+0.001199	test-mae:1.379183+0.011525 
    [86]	train-mae:1.369180+0.001197	test-mae:1.378965+0.011374 
    [87]	train-mae:1.369020+0.001204	test-mae:1.378902+0.011349 
    [88]	train-mae:1.368942+0.001217	test-mae:1.378815+0.011300 
    [89]	train-mae:1.368856+0.001276	test-mae:1.378815+0.011492 
    [90]	train-mae:1.368618+0.001223	test-mae:1.378598+0.011158 
    [91]	train-mae:1.368457+0.001186	test-mae:1.378502+0.011175 
    [92]	train-mae:1.368363+0.001203	test-mae:1.378503+0.011192 
    [93]	train-mae:1.368254+0.001245	test-mae:1.378476+0.011179 
    [94]	train-mae:1.368154+0.001274	test-mae:1.378407+0.011217 
    [95]	train-mae:1.368082+0.001187	test-mae:1.378324+0.011237 
    [96]	train-mae:1.367926+0.001165	test-mae:1.378186+0.011303 
    [97]	train-mae:1.367740+0.001062	test-mae:1.378108+0.011295 
    [98]	train-mae:1.367675+0.001023	test-mae:1.378093+0.011367 
    [99]	train-mae:1.367498+0.001008	test-mae:1.378002+0.011274 
    [100]	train-mae:1.367321+0.001077	test-mae:1.377856+0.011254 
    [101]	train-mae:1.367283+0.001094	test-mae:1.377912+0.011218 
    [102]	train-mae:1.367122+0.001054	test-mae:1.377886+0.011253 
    [103]	train-mae:1.366968+0.001066	test-mae:1.377719+0.011194 
    [104]	train-mae:1.366832+0.001136	test-mae:1.377685+0.011149 
    [105]	train-mae:1.366696+0.001152	test-mae:1.377573+0.011030 
    [106]	train-mae:1.366583+0.001112	test-mae:1.377515+0.011013 
    [107]	train-mae:1.366515+0.001139	test-mae:1.377499+0.011049 
    [108]	train-mae:1.366433+0.001124	test-mae:1.377505+0.011145 
    [109]	train-mae:1.366370+0.001148	test-mae:1.377468+0.011031 
    [110]	train-mae:1.366223+0.001145	test-mae:1.377383+0.011073 
    [111]	train-mae:1.366135+0.001153	test-mae:1.377394+0.011027 
    [112]	train-mae:1.366003+0.001173	test-mae:1.377400+0.010978 
    [113]	train-mae:1.365900+0.001177	test-mae:1.377365+0.010987 
    [114]	train-mae:1.365771+0.001124	test-mae:1.377286+0.010868 
    [115]	train-mae:1.365683+0.001128	test-mae:1.377268+0.010862 
    [116]	train-mae:1.365628+0.001131	test-mae:1.377230+0.010878 
    [117]	train-mae:1.365533+0.001142	test-mae:1.377207+0.010840 
    [118]	train-mae:1.365420+0.001155	test-mae:1.377192+0.010841 
    [119]	train-mae:1.365290+0.001192	test-mae:1.377147+0.010861 
    [120]	train-mae:1.365197+0.001200	test-mae:1.377066+0.010824 
    [121]	train-mae:1.365091+0.001173	test-mae:1.377088+0.010813 
    [122]	train-mae:1.365036+0.001189	test-mae:1.377036+0.010780 
    [123]	train-mae:1.364877+0.001188	test-mae:1.376944+0.010811 
    [124]	train-mae:1.364765+0.001195	test-mae:1.376843+0.010713 
    [125]	train-mae:1.364642+0.001198	test-mae:1.376835+0.010684 
    [126]	train-mae:1.364506+0.001191	test-mae:1.376746+0.010649 
    [127]	train-mae:1.364374+0.001157	test-mae:1.376670+0.010682 
    [128]	train-mae:1.364261+0.001146	test-mae:1.376531+0.010755 
    [129]	train-mae:1.364146+0.001120	test-mae:1.376446+0.010825 
    [130]	train-mae:1.364062+0.001119	test-mae:1.376488+0.010856 
    [131]	train-mae:1.363981+0.001141	test-mae:1.376457+0.010863 
    [132]	train-mae:1.363838+0.001106	test-mae:1.376358+0.010858 
    [133]	train-mae:1.363724+0.001176	test-mae:1.376293+0.010894 
    [134]	train-mae:1.363581+0.001165	test-mae:1.376184+0.010817 
    [135]	train-mae:1.363528+0.001208	test-mae:1.376213+0.010875 
    [136]	train-mae:1.363418+0.001159	test-mae:1.376148+0.010850 
    [137]	train-mae:1.363319+0.001138	test-mae:1.376092+0.010817 
    [138]	train-mae:1.363182+0.001098	test-mae:1.376052+0.010818 
    [139]	train-mae:1.363098+0.001130	test-mae:1.376070+0.010746 
    [140]	train-mae:1.363014+0.001161	test-mae:1.376039+0.010778 
    [141]	train-mae:1.362893+0.001144	test-mae:1.375980+0.010745 
    [142]	train-mae:1.362801+0.001172	test-mae:1.375999+0.010683 
    [143]	train-mae:1.362665+0.001132	test-mae:1.375888+0.010638 
    [144]	train-mae:1.362586+0.001139	test-mae:1.375932+0.010662 
    [145]	train-mae:1.362484+0.001152	test-mae:1.375923+0.010670 
    [146]	train-mae:1.362375+0.001137	test-mae:1.375922+0.010644 
    [147]	train-mae:1.362226+0.001162	test-mae:1.375873+0.010681 
    [148]	train-mae:1.362119+0.001157	test-mae:1.375832+0.010694 
    [149]	train-mae:1.362035+0.001157	test-mae:1.375827+0.010700 
    [150]	train-mae:1.361948+0.001145	test-mae:1.375815+0.010696 
    [151]	train-mae:1.361838+0.001131	test-mae:1.375799+0.010750 
    [152]	train-mae:1.361765+0.001149	test-mae:1.375865+0.010725 
    [153]	train-mae:1.361662+0.001125	test-mae:1.375873+0.010730 
    [154]	train-mae:1.361575+0.001125	test-mae:1.375809+0.010715 
    [155]	train-mae:1.361447+0.001187	test-mae:1.375703+0.010702 
    [156]	train-mae:1.361363+0.001211	test-mae:1.375701+0.010700 
    [157]	train-mae:1.361276+0.001207	test-mae:1.375662+0.010743 
    [158]	train-mae:1.361163+0.001234	test-mae:1.375630+0.010771 
    [159]	train-mae:1.361040+0.001225	test-mae:1.375589+0.010728 
    [160]	train-mae:1.360953+0.001250	test-mae:1.375601+0.010738 
    [161]	train-mae:1.360848+0.001242	test-mae:1.375646+0.010683 
    [162]	train-mae:1.360738+0.001192	test-mae:1.375622+0.010639 
    [163]	train-mae:1.360662+0.001161	test-mae:1.375588+0.010627 
    [164]	train-mae:1.360534+0.001170	test-mae:1.375551+0.010637 
    [165]	train-mae:1.360449+0.001178	test-mae:1.375563+0.010674 
    [166]	train-mae:1.360360+0.001161	test-mae:1.375589+0.010655 
    [167]	train-mae:1.360263+0.001112	test-mae:1.375590+0.010650 
    [168]	train-mae:1.360192+0.001104	test-mae:1.375598+0.010642 
    [169]	train-mae:1.360106+0.001086	test-mae:1.375632+0.010594 
    [170]	train-mae:1.360021+0.001077	test-mae:1.375682+0.010624 
    [171]	train-mae:1.359927+0.001041	test-mae:1.375601+0.010618 
    [172]	train-mae:1.359859+0.001026	test-mae:1.375657+0.010649 
    [173]	train-mae:1.359771+0.001012	test-mae:1.375622+0.010687 
    [174]	train-mae:1.359706+0.001013	test-mae:1.375630+0.010686 
    [175]	train-mae:1.359623+0.001002	test-mae:1.375595+0.010744 
    [176]	train-mae:1.359493+0.001020	test-mae:1.375560+0.010701 
    [177]	train-mae:1.359415+0.001029	test-mae:1.375567+0.010724 
    [178]	train-mae:1.359323+0.001020	test-mae:1.375492+0.010763 
    [179]	train-mae:1.359239+0.001003	test-mae:1.375473+0.010730 
    [180]	train-mae:1.359151+0.000998	test-mae:1.375507+0.010747 
    [181]	train-mae:1.359068+0.000992	test-mae:1.375524+0.010643 
    [182]	train-mae:1.358972+0.000986	test-mae:1.375526+0.010670 
    [183]	train-mae:1.358864+0.000965	test-mae:1.375506+0.010641 
    [184]	train-mae:1.358786+0.000992	test-mae:1.375453+0.010591 
    [185]	train-mae:1.358704+0.000997	test-mae:1.375414+0.010543 
    [186]	train-mae:1.358595+0.000984	test-mae:1.375373+0.010536 
    [187]	train-mae:1.358519+0.000979	test-mae:1.375378+0.010517 
    [188]	train-mae:1.358454+0.000988	test-mae:1.375347+0.010484 
    [189]	train-mae:1.358376+0.001005	test-mae:1.375338+0.010486 
    [190]	train-mae:1.358301+0.001021	test-mae:1.375342+0.010472 
    [191]	train-mae:1.358200+0.000986	test-mae:1.375327+0.010438 
    [192]	train-mae:1.358123+0.001021	test-mae:1.375312+0.010452 
    [193]	train-mae:1.358048+0.001017	test-mae:1.375242+0.010408 
    [194]	train-mae:1.357983+0.000991	test-mae:1.375220+0.010440 
    [195]	train-mae:1.357921+0.000994	test-mae:1.375235+0.010401 
    [196]	train-mae:1.357844+0.001008	test-mae:1.375231+0.010424 
    [197]	train-mae:1.357742+0.000995	test-mae:1.375194+0.010347 
    [198]	train-mae:1.357658+0.001008	test-mae:1.375197+0.010321 
    [199]	train-mae:1.357580+0.001007	test-mae:1.375169+0.010290 
    [200]	train-mae:1.357513+0.000981	test-mae:1.375147+0.010289 
    [201]	train-mae:1.357413+0.000952	test-mae:1.375137+0.010331 
    [202]	train-mae:1.357349+0.000963	test-mae:1.375138+0.010290 
    [203]	train-mae:1.357266+0.000979	test-mae:1.375091+0.010308 
    [204]	train-mae:1.357198+0.000991	test-mae:1.375131+0.010317 
    [205]	train-mae:1.357117+0.000983	test-mae:1.375088+0.010365 
    [206]	train-mae:1.357053+0.000976	test-mae:1.375081+0.010374 
    [207]	train-mae:1.356992+0.000984	test-mae:1.375118+0.010466 
    [208]	train-mae:1.356954+0.000988	test-mae:1.375144+0.010464 
    [209]	train-mae:1.356867+0.000974	test-mae:1.375118+0.010465 
    [210]	train-mae:1.356793+0.000960	test-mae:1.375147+0.010420 
    [211]	train-mae:1.356681+0.000956	test-mae:1.375087+0.010472 
    [212]	train-mae:1.356608+0.000927	test-mae:1.375090+0.010440 
    [213]	train-mae:1.356512+0.000931	test-mae:1.375127+0.010480 
    [214]	train-mae:1.356453+0.000933	test-mae:1.375132+0.010496 
    [215]	train-mae:1.356368+0.000938	test-mae:1.375090+0.010550 
    [216]	train-mae:1.356278+0.000933	test-mae:1.375044+0.010514 
    [217]	train-mae:1.356212+0.000919	test-mae:1.375055+0.010534 
    [218]	train-mae:1.356161+0.000916	test-mae:1.375063+0.010532 
    [219]	train-mae:1.356091+0.000903	test-mae:1.375072+0.010489 
    [220]	train-mae:1.356015+0.000924	test-mae:1.375072+0.010516 
    [221]	train-mae:1.355940+0.000918	test-mae:1.375073+0.010530 
    [222]	train-mae:1.355860+0.000929	test-mae:1.375084+0.010506 
    [223]	train-mae:1.355790+0.000916	test-mae:1.375049+0.010557 
    [224]	train-mae:1.355716+0.000934	test-mae:1.375034+0.010541 
    [225]	train-mae:1.355636+0.000955	test-mae:1.375018+0.010540 
    [226]	train-mae:1.355537+0.000933	test-mae:1.375091+0.010544 
    [227]	train-mae:1.355470+0.000950	test-mae:1.375122+0.010564 
    [228]	train-mae:1.355399+0.000974	test-mae:1.375126+0.010588 
    [229]	train-mae:1.355308+0.000977	test-mae:1.375145+0.010558 
    [230]	train-mae:1.355232+0.000973	test-mae:1.375116+0.010539 
    [231]	train-mae:1.355136+0.000948	test-mae:1.375061+0.010522 
    [232]	train-mae:1.355039+0.000938	test-mae:1.375065+0.010516 
    [233]	train-mae:1.354964+0.000924	test-mae:1.375065+0.010505 
    [234]	train-mae:1.354870+0.000939	test-mae:1.375030+0.010477 
    [235]	train-mae:1.354773+0.000930	test-mae:1.375011+0.010426 
    [236]	train-mae:1.354712+0.000937	test-mae:1.374960+0.010439 
    [237]	train-mae:1.354658+0.000929	test-mae:1.374998+0.010452 
    [238]	train-mae:1.354573+0.000928	test-mae:1.374984+0.010395 
    [239]	train-mae:1.354486+0.000927	test-mae:1.375036+0.010389 
    [240]	train-mae:1.354394+0.000955	test-mae:1.375030+0.010349 
    [241]	train-mae:1.354303+0.000950	test-mae:1.375004+0.010339 
    [242]	train-mae:1.354196+0.000892	test-mae:1.374979+0.010369 
    [243]	train-mae:1.354136+0.000891	test-mae:1.374997+0.010335 
    [244]	train-mae:1.354070+0.000878	test-mae:1.375007+0.010346 
    [245]	train-mae:1.354001+0.000855	test-mae:1.375032+0.010361 
    [246]	train-mae:1.353924+0.000873	test-mae:1.374998+0.010327 
    [247]	train-mae:1.353846+0.000883	test-mae:1.374948+0.010295 
    [248]	train-mae:1.353767+0.000918	test-mae:1.374905+0.010289 
    [249]	train-mae:1.353678+0.000914	test-mae:1.374891+0.010341 
    [250]	train-mae:1.353617+0.000910	test-mae:1.374923+0.010379 
    [251]	train-mae:1.353532+0.000904	test-mae:1.374937+0.010431 
    [252]	train-mae:1.353435+0.000918	test-mae:1.374902+0.010405 
    [253]	train-mae:1.353359+0.000930	test-mae:1.374865+0.010456 
    [254]	train-mae:1.353274+0.000908	test-mae:1.374831+0.010446 
    [255]	train-mae:1.353172+0.000898	test-mae:1.374765+0.010466 
    [256]	train-mae:1.353102+0.000888	test-mae:1.374745+0.010414 
    [257]	train-mae:1.353030+0.000878	test-mae:1.374742+0.010406 
    [258]	train-mae:1.352971+0.000891	test-mae:1.374776+0.010425 
    [259]	train-mae:1.352892+0.000869	test-mae:1.374723+0.010416 
    [260]	train-mae:1.352793+0.000863	test-mae:1.374719+0.010391 
    [261]	train-mae:1.352735+0.000858	test-mae:1.374661+0.010332 
    [262]	train-mae:1.352657+0.000832	test-mae:1.374700+0.010358 
    [263]	train-mae:1.352579+0.000814	test-mae:1.374711+0.010411 
    [264]	train-mae:1.352507+0.000794	test-mae:1.374680+0.010400 
    [265]	train-mae:1.352446+0.000790	test-mae:1.374710+0.010372 
    [266]	train-mae:1.352397+0.000792	test-mae:1.374725+0.010385 
    [267]	train-mae:1.352344+0.000794	test-mae:1.374752+0.010377 
    [268]	train-mae:1.352281+0.000793	test-mae:1.374735+0.010387 
    [269]	train-mae:1.352225+0.000800	test-mae:1.374750+0.010415 
    [270]	train-mae:1.352155+0.000800	test-mae:1.374740+0.010396 
    [271]	train-mae:1.352101+0.000795	test-mae:1.374723+0.010370 
    [272]	train-mae:1.352036+0.000792	test-mae:1.374762+0.010369 
    [273]	train-mae:1.351979+0.000793	test-mae:1.374795+0.010345 
    [274]	train-mae:1.351913+0.000803	test-mae:1.374819+0.010342 
    [275]	train-mae:1.351833+0.000788	test-mae:1.374798+0.010359 
    [276]	train-mae:1.351732+0.000802	test-mae:1.374780+0.010366 
    [277]	train-mae:1.351640+0.000806	test-mae:1.374771+0.010328 
    Stopping. Best iteration:
    [261]	train-mae:1.352735+0.000858	test-mae:1.374661+0.010332
    


**Use 261 rounds.**


```R
set.seed(155)
xgb4 = xgboost(params = list(eta = .3, 
                            gamma = 0,
                            subsample = .8,
                            colsample_bytree = .9, 
                            reg_alpha = .08,
                            reg_lambda = 1),
              max_depth = 3,
              min_child_weight = 16,
              data = train.xgb, 
              nrounds = 261,
              objective = 'reg:absoluteerror')

```

    [1]	train-mae:2.224217 
    [2]	train-mae:2.133178 
    [3]	train-mae:2.058188 
    [4]	train-mae:1.966525 
    [5]	train-mae:1.890085 
    [6]	train-mae:1.821229 
    [7]	train-mae:1.757355 
    [8]	train-mae:1.697608 
    [9]	train-mae:1.647538 
    [10]	train-mae:1.602516 
    [11]	train-mae:1.562911 
    [12]	train-mae:1.531956 
    [13]	train-mae:1.510482 
    [14]	train-mae:1.500075 
    [15]	train-mae:1.492045 
    [16]	train-mae:1.475694 
    [17]	train-mae:1.462096 
    [18]	train-mae:1.452926 
    [19]	train-mae:1.443468 
    [20]	train-mae:1.435148 
    [21]	train-mae:1.429704 
    [22]	train-mae:1.425044 
    [23]	train-mae:1.421211 
    [24]	train-mae:1.417648 
    [25]	train-mae:1.414832 
    [26]	train-mae:1.413066 
    [27]	train-mae:1.409969 
    [28]	train-mae:1.407814 
    [29]	train-mae:1.405923 
    [30]	train-mae:1.404213 
    [31]	train-mae:1.402330 
    [32]	train-mae:1.400999 
    [33]	train-mae:1.400098 
    [34]	train-mae:1.398768 
    [35]	train-mae:1.397129 
    [36]	train-mae:1.396172 
    [37]	train-mae:1.394995 
    [38]	train-mae:1.393692 
    [39]	train-mae:1.392451 
    [40]	train-mae:1.391474 
    [41]	train-mae:1.389928 
    [42]	train-mae:1.388447 
    [43]	train-mae:1.387321 
    [44]	train-mae:1.386350 
    [45]	train-mae:1.385831 
    [46]	train-mae:1.385273 
    [47]	train-mae:1.384522 
    [48]	train-mae:1.383295 
    [49]	train-mae:1.383040 
    [50]	train-mae:1.382479 
    [51]	train-mae:1.382074 
    [52]	train-mae:1.381328 
    [53]	train-mae:1.380898 
    [54]	train-mae:1.380041 
    [55]	train-mae:1.379791 
    [56]	train-mae:1.379149 
    [57]	train-mae:1.378446 
    [58]	train-mae:1.378107 
    [59]	train-mae:1.377887 
    [60]	train-mae:1.377656 
    [61]	train-mae:1.377067 
    [62]	train-mae:1.376748 
    [63]	train-mae:1.376484 
    [64]	train-mae:1.376117 
    [65]	train-mae:1.375709 
    [66]	train-mae:1.375608 
    [67]	train-mae:1.375532 
    [68]	train-mae:1.375015 
    [69]	train-mae:1.374670 
    [70]	train-mae:1.374200 
    [71]	train-mae:1.374192 
    [72]	train-mae:1.373805 
    [73]	train-mae:1.373813 
    [74]	train-mae:1.373704 
    [75]	train-mae:1.373572 
    [76]	train-mae:1.373574 
    [77]	train-mae:1.373449 
    [78]	train-mae:1.373414 
    [79]	train-mae:1.373069 
    [80]	train-mae:1.373089 
    [81]	train-mae:1.372800 
    [82]	train-mae:1.372539 
    [83]	train-mae:1.372297 
    [84]	train-mae:1.372098 
    [85]	train-mae:1.371993 
    [86]	train-mae:1.371717 
    [87]	train-mae:1.371275 
    [88]	train-mae:1.371146 
    [89]	train-mae:1.371031 
    [90]	train-mae:1.370748 
    [91]	train-mae:1.370596 
    [92]	train-mae:1.370342 
    [93]	train-mae:1.369734 
    [94]	train-mae:1.369567 
    [95]	train-mae:1.369270 
    [96]	train-mae:1.368952 
    [97]	train-mae:1.368680 
    [98]	train-mae:1.368468 
    [99]	train-mae:1.368306 
    [100]	train-mae:1.368158 
    [101]	train-mae:1.367797 
    [102]	train-mae:1.367642 
    [103]	train-mae:1.367508 
    [104]	train-mae:1.367379 
    [105]	train-mae:1.367248 
    [106]	train-mae:1.367102 
    [107]	train-mae:1.366886 
    [108]	train-mae:1.366722 
    [109]	train-mae:1.366775 
    [110]	train-mae:1.366611 
    [111]	train-mae:1.366635 
    [112]	train-mae:1.366505 
    [113]	train-mae:1.366373 
    [114]	train-mae:1.366148 
    [115]	train-mae:1.365964 
    [116]	train-mae:1.365878 
    [117]	train-mae:1.365674 
    [118]	train-mae:1.365533 
    [119]	train-mae:1.365492 
    [120]	train-mae:1.365366 
    [121]	train-mae:1.365269 
    [122]	train-mae:1.365151 
    [123]	train-mae:1.364990 
    [124]	train-mae:1.364607 
    [125]	train-mae:1.364391 
    [126]	train-mae:1.364138 
    [127]	train-mae:1.363988 
    [128]	train-mae:1.363928 
    [129]	train-mae:1.363878 
    [130]	train-mae:1.363762 
    [131]	train-mae:1.363636 
    [132]	train-mae:1.363544 
    [133]	train-mae:1.363352 
    [134]	train-mae:1.363306 
    [135]	train-mae:1.363060 
    [136]	train-mae:1.363049 
    [137]	train-mae:1.362837 
    [138]	train-mae:1.362834 
    [139]	train-mae:1.362802 
    [140]	train-mae:1.362613 
    [141]	train-mae:1.362544 
    [142]	train-mae:1.362438 
    [143]	train-mae:1.362372 
    [144]	train-mae:1.362370 
    [145]	train-mae:1.362326 
    [146]	train-mae:1.362250 
    [147]	train-mae:1.362199 
    [148]	train-mae:1.362156 
    [149]	train-mae:1.362081 
    [150]	train-mae:1.361981 
    [151]	train-mae:1.361950 
    [152]	train-mae:1.361891 
    [153]	train-mae:1.361837 
    [154]	train-mae:1.361825 
    [155]	train-mae:1.361767 
    [156]	train-mae:1.361740 
    [157]	train-mae:1.361658 
    [158]	train-mae:1.361576 
    [159]	train-mae:1.361516 
    [160]	train-mae:1.361461 
    [161]	train-mae:1.361399 
    [162]	train-mae:1.361364 
    [163]	train-mae:1.361346 
    [164]	train-mae:1.361242 
    [165]	train-mae:1.361178 
    [166]	train-mae:1.361118 
    [167]	train-mae:1.361016 
    [168]	train-mae:1.360946 
    [169]	train-mae:1.360838 
    [170]	train-mae:1.360705 
    [171]	train-mae:1.360664 
    [172]	train-mae:1.360587 
    [173]	train-mae:1.360468 
    [174]	train-mae:1.360336 
    [175]	train-mae:1.360282 
    [176]	train-mae:1.360211 
    [177]	train-mae:1.360145 
    [178]	train-mae:1.360112 
    [179]	train-mae:1.360079 
    [180]	train-mae:1.360067 
    [181]	train-mae:1.360022 
    [182]	train-mae:1.359888 
    [183]	train-mae:1.359811 
    [184]	train-mae:1.359759 
    [185]	train-mae:1.359689 
    [186]	train-mae:1.359623 
    [187]	train-mae:1.359617 
    [188]	train-mae:1.359500 
    [189]	train-mae:1.359424 
    [190]	train-mae:1.359388 
    [191]	train-mae:1.359378 
    [192]	train-mae:1.359322 
    [193]	train-mae:1.359214 
    [194]	train-mae:1.359144 
    [195]	train-mae:1.359124 
    [196]	train-mae:1.359041 
    [197]	train-mae:1.358982 
    [198]	train-mae:1.358957 
    [199]	train-mae:1.358915 
    [200]	train-mae:1.358838 
    [201]	train-mae:1.358755 
    [202]	train-mae:1.358549 
    [203]	train-mae:1.358470 
    [204]	train-mae:1.358370 
    [205]	train-mae:1.358249 
    [206]	train-mae:1.358170 
    [207]	train-mae:1.358149 
    [208]	train-mae:1.358015 
    [209]	train-mae:1.358020 
    [210]	train-mae:1.357947 
    [211]	train-mae:1.357938 
    [212]	train-mae:1.357907 
    [213]	train-mae:1.357843 
    [214]	train-mae:1.357779 
    [215]	train-mae:1.357707 
    [216]	train-mae:1.357684 
    [217]	train-mae:1.357570 
    [218]	train-mae:1.357478 
    [219]	train-mae:1.357426 
    [220]	train-mae:1.357342 
    [221]	train-mae:1.357272 
    [222]	train-mae:1.357176 
    [223]	train-mae:1.357114 
    [224]	train-mae:1.357044 
    [225]	train-mae:1.356968 
    [226]	train-mae:1.356962 
    [227]	train-mae:1.356811 
    [228]	train-mae:1.356740 
    [229]	train-mae:1.356645 
    [230]	train-mae:1.356514 
    [231]	train-mae:1.356474 
    [232]	train-mae:1.356389 
    [233]	train-mae:1.356331 
    [234]	train-mae:1.356311 
    [235]	train-mae:1.356167 
    [236]	train-mae:1.356138 
    [237]	train-mae:1.356087 
    [238]	train-mae:1.356064 
    [239]	train-mae:1.356021 
    [240]	train-mae:1.355872 
    [241]	train-mae:1.355871 
    [242]	train-mae:1.355834 
    [243]	train-mae:1.355835 
    [244]	train-mae:1.355791 
    [245]	train-mae:1.355729 
    [246]	train-mae:1.355702 
    [247]	train-mae:1.355690 
    [248]	train-mae:1.355608 
    [249]	train-mae:1.355512 
    [250]	train-mae:1.355438 
    [251]	train-mae:1.355399 
    [252]	train-mae:1.355343 
    [253]	train-mae:1.355293 
    [254]	train-mae:1.355250 
    [255]	train-mae:1.355185 
    [256]	train-mae:1.355118 
    [257]	train-mae:1.355067 
    [258]	train-mae:1.354949 
    [259]	train-mae:1.354917 
    [260]	train-mae:1.354833 
    [261]	train-mae:1.354822 


**Plot variable importance.**


```R
xgb4.imp = xgb.importance(colnames(train.xgb), model = xgb4)
xgb.plot.importance(xgb4.imp)
```


    
![png](crab-xgboost_files/crab-xgboost_49_0.png)
    


**Shell weight is by far the most important variable in this model.**

**Predict with xgb4.  It helps to round the predictions to the nearest integer.  (See the Discussion board for this competition for more info.)**


```R
xgb4.age = round(predict(xgb4, test.xgb))
range(train$Age)
range(xgb4.age)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>1</li><li>29</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>3</li><li>18</li></ol>




```R
xgb4.guess = cbind.data.frame(test[1], xgb4.age)
colnames(xgb4.guess) = c('id', 'Age')
write.csv(xgb4.guess, 'submission.csv', row.names = F)
```
