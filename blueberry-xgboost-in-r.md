<a href="https://www.kaggle.com/code/jeremyhaakenson/blueberry-xgboost-in-r?scriptVersionId=285331855" target="_blank"><img align="left" alt="Kaggle" title="Open in Kaggle" src="https://kaggle.com/static/images/open-in-kaggle.svg"></a>

Load packages.


```R
library(dplyr)
library(xgboost)
```

    
    Attaching package: ‘dplyr’
    
    
    The following objects are masked from ‘package:stats’:
    
        filter, lag
    
    
    The following objects are masked from ‘package:base’:
    
        intersect, setdiff, setequal, union
    
    
    
    Attaching package: ‘xgboost’
    
    
    The following object is masked from ‘package:dplyr’:
    
        slice
    
    


Load data.


```R
train = read.csv("/kaggle/input/playground-series-s3e14/train.csv")
test = read.csv("/kaggle/input/playground-series-s3e14/test.csv")
```

Remove highly correlated variables.


```R
train.new = train[, c(1:6, 11, 13, 15:18)]
test.new = test[, c(1:6, 11, 13, 15:17)] 
```

Scale the data.


```R
train.scale = scale(train.new[, 2:11])
train.scale = cbind.data.frame(train$id, train.scale, train$yield)
colnames(train.scale)[1] = 'id'
colnames(train.scale)[12] = 'yield'
test.scale = scale(test.new[, 2:11])
test.scale = cbind.data.frame(test$id, test.scale)
colnames(test.scale)[1] = 'id'
```

**Feature Engineering**

Add a variable for the total number of bees.



```R
train.scale$TotBees = scale(train.scale$honeybee + train.scale$bumbles +
                              train.scale$andrena +  train.scale$osmia)
plot(train.scale$TotBees)
```


    
![png](blueberry-xgboost-in-r_files/blueberry-xgboost-in-r_10_0.png)
    


Log transform the new variable.


```R
train.scale$TotBees = log(train.scale$TotBees + 6)
plot(train.scale$TotBees)
test.scale$TotBees = scale(test.scale$honeybee + test.scale$bumbles +
                              test.scale$andrena +  test.scale$osmia)
test.scale$TotBees = log(test.scale$TotBees + 6)
```


    
![png](blueberry-xgboost-in-r_files/blueberry-xgboost-in-r_12_0.png)
    


Make a variable of honeybee * bumbles.


```R
train.scale$honbum = scale(train.scale$honeybee * train.scale$bumbles)
test.scale$honbum = scale(test.scale$honeybee * test.scale$bumbles)
```

Remove honeybee since it is highly correlated with honbum.


```R
train.scale['honeybee'] = NULL
test.scale['honeybee'] = NULL
```

Remove duplicates


```R
train.nodup = train.scale %>%
  distinct(clonesize, bumbles, andrena, osmia, MinOfLowerTRange, 
           RainingDays, fruitset, TotBees, honbum, fruitmass, seeds,
           .keep_all = T)

```

Look for outliers in the target variable.


```R
boxplot(train.nodup$yield)
```


    
![png](blueberry-xgboost-in-r_files/blueberry-xgboost-in-r_20_0.png)
    


There is 1 low outlier value in yield. Change yields of 1945.53061 to 2379.90521 (the next lowest value).


```R
train.out = train.nodup
min(train.out$yield)
train.out$yield[train.out$yield == 1945.53061] = 2379.90521   
min(train.out$yield)
```


1945.53061



2379.90521


Check dataframes to make sure everything looks OK.


```R
head(train.out)
head(test.scale)
```


<table class="dataframe">
<caption>A data.frame: 6 × 13</caption>
<thead>
	<tr><th></th><th scope=col>id</th><th scope=col>clonesize</th><th scope=col>bumbles</th><th scope=col>andrena</th><th scope=col>osmia</th><th scope=col>MinOfLowerTRange</th><th scope=col>RainingDays</th><th scope=col>fruitset</th><th scope=col>fruitmass</th><th scope=col>seeds</th><th scope=col>yield</th><th scope=col>TotBees</th><th scope=col>honbum</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl[,1]&gt;</th><th scope=col>&lt;dbl[,1]&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0</td><td> 0.8029024</td><td>-0.6136444</td><td>1.73732964</td><td>-0.6620986</td><td>-1.3646489</td><td>0.4579968</td><td>-1.0449040</td><td>-0.7832340</td><td>-0.9188746</td><td>4476.811</td><td>1.849641</td><td>-0.04455279</td></tr>
	<tr><th scope=row>2</th><td>1</td><td> 0.8029024</td><td>-0.6136444</td><td>0.04945219</td><td>-0.6620986</td><td>-1.3646489</td><td>0.4579968</td><td>-0.7774287</td><td>-0.6615667</td><td>-0.5722113</td><td>5548.122</td><td>1.717647</td><td>-0.04455279</td></tr>
	<tr><th scope=row>3</th><td>2</td><td>-1.0924123</td><td>-0.6136444</td><td>0.92714847</td><td> 0.2698744</td><td> 0.4191840</td><td>0.4579968</td><td> 0.6746365</td><td> 0.6561355</td><td> 0.5400109</td><td>6869.778</td><td>1.807024</td><td> 0.06670053</td></tr>
	<tr><th scope=row>4</th><td>3</td><td>-1.0924123</td><td>-0.6136444</td><td>0.92714847</td><td>-0.6620986</td><td>-0.5196754</td><td>0.4579968</td><td> 0.8500596</td><td> 0.8528038</td><td> 0.8192855</td><td>6880.776</td><td>1.733110</td><td> 0.06670053</td></tr>
	<tr><th scope=row>5</th><td>4</td><td> 0.8029024</td><td>-0.6136444</td><td>0.92714847</td><td> 0.2698744</td><td>-0.5196754</td><td>0.4579968</td><td> 1.0342272</td><td> 1.2855857</td><td> 1.0715626</td><td>7479.934</td><td>1.858524</td><td>-0.04455279</td></tr>
	<tr><th scope=row>6</th><td>5</td><td> 0.8029024</td><td>-0.6136444</td><td>0.92714847</td><td> 1.1301571</td><td> 1.3580435</td><td>1.3158076</td><td> 0.8401510</td><td> 1.0205615</td><td> 1.0890535</td><td>7267.283</td><td>1.919117</td><td>-0.04455279</td></tr>
</tbody>
</table>




<table class="dataframe">
<caption>A data.frame: 6 × 12</caption>
<thead>
	<tr><th></th><th scope=col>id</th><th scope=col>clonesize</th><th scope=col>bumbles</th><th scope=col>andrena</th><th scope=col>osmia</th><th scope=col>MinOfLowerTRange</th><th scope=col>RainingDays</th><th scope=col>fruitset</th><th scope=col>fruitmass</th><th scope=col>seeds</th><th scope=col>TotBees</th><th scope=col>honbum</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl[,1]&gt;</th><th scope=col>&lt;dbl[,1]&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>15289</td><td> 0.7894643</td><td>-0.6064728</td><td>-1.6287649</td><td>-2.4533852</td><td>0.4143539</td><td> 0.4662653</td><td>-1.3712508</td><td>-1.03245886</td><td>-1.1795811</td><td>1.296934</td><td> 0.05212116</td></tr>
	<tr><th scope=row>2</th><td>15290</td><td>-1.1068126</td><td>-0.6064728</td><td> 1.7458194</td><td> 0.2628944</td><td>1.3523573</td><td>-1.4861842</td><td>-0.1841978</td><td>-0.09389294</td><td> 0.1767731</td><td>1.868991</td><td> 0.05212116</td></tr>
	<tr><th scope=row>3</th><td>15291</td><td>-1.1068126</td><td>-0.6064728</td><td> 0.9359191</td><td> 0.2628944</td><td>0.4143539</td><td>-0.2128476</td><td> 1.0918685</td><td> 1.09872641</td><td> 0.9704992</td><td>1.809319</td><td> 0.05212116</td></tr>
	<tr><th scope=row>4</th><td>15292</td><td> 0.7894643</td><td> 1.5650237</td><td>-0.7513730</td><td> 0.2628944</td><td>0.4143539</td><td>-0.2128476</td><td>-0.9208593</td><td>-0.63415985</td><td>-0.7513294</td><td>1.891291</td><td> 0.10390840</td></tr>
	<tr><th scope=row>5</th><td>15293</td><td> 2.6857413</td><td>-0.6064728</td><td>-1.6287649</td><td>-2.4533852</td><td>1.3523573</td><td> 0.4662653</td><td>-1.8848766</td><td>-1.55137255</td><td>-1.6364472</td><td>1.447479</td><td>-0.13536435</td></tr>
	<tr><th scope=row>6</th><td>15294</td><td>-1.1068126</td><td>-0.6064728</td><td> 1.7458194</td><td> 1.1206670</td><td>1.3523573</td><td> 1.3151564</td><td>-0.6679203</td><td>-0.28941424</td><td>-0.8263391</td><td>1.928533</td><td> 0.05212116</td></tr>
</tbody>
</table>



Put data in XGBoost format.


```R
train.xgb <- xgb.DMatrix(label = train.out$yield, data = as.matrix(train.out[-11]))
test.xgb = as.matrix(test.scale)
```

Determine number of rounds to use after optimizing parameters (not shown).


```R
set.seed(12)
xgb8 = xgb.cv(params = list(eta = .006, min_child_weight = 3, 
                            subsample = .7, reg_lambda = .5), 
              data = train.xgb, nrounds = 1200, 
              max_depth = 5, 
              early_stopping_rounds = 16, nfold = 9, metrics = c('mae'))
```

    [1]	train-mae:5989.120647+3.327733	test-mae:5989.133160+26.590664 
    Multiple eval metrics are present. Will use test_mae for early stopping.
    Will train until test_mae hasn't improved in 16 rounds.
    
    [2]	train-mae:5953.214890+3.310090	test-mae:5953.231969+26.454257 
    [3]	train-mae:5917.531613+3.291346	test-mae:5917.540654+26.315997 
    [4]	train-mae:5882.049721+3.265893	test-mae:5882.057984+26.186866 
    [5]	train-mae:5846.775440+3.250454	test-mae:5846.778905+26.057060 
    [6]	train-mae:5811.726462+3.238963	test-mae:5811.735551+25.901057 
    [7]	train-mae:5776.897379+3.212653	test-mae:5776.910768+25.777513 
    [8]	train-mae:5742.253299+3.208676	test-mae:5742.276547+25.625253 
    [9]	train-mae:5707.831991+3.188713	test-mae:5707.850296+25.498641 
    [10]	train-mae:5673.611438+3.180111	test-mae:5673.634601+25.360811 
    [11]	train-mae:5639.613246+3.170735	test-mae:5639.636720+25.228622 
    [12]	train-mae:5605.794418+3.156663	test-mae:5605.819622+25.102545 
    [13]	train-mae:5572.184772+3.144185	test-mae:5572.206598+24.965346 
    [14]	train-mae:5538.787200+3.128279	test-mae:5538.805540+24.834156 
    [15]	train-mae:5505.581855+3.116083	test-mae:5505.603172+24.700125 
    [16]	train-mae:5472.583440+3.104703	test-mae:5472.603595+24.570655 
    [17]	train-mae:5439.777717+3.096557	test-mae:5439.806379+24.446782 
    [18]	train-mae:5407.169583+3.081122	test-mae:5407.199586+24.333594 
    [19]	train-mae:5374.754275+3.049394	test-mae:5374.780493+24.228065 
    [20]	train-mae:5342.524090+3.051037	test-mae:5342.548266+24.091702 
    [21]	train-mae:5310.503866+3.025023	test-mae:5310.523836+23.995252 
    [22]	train-mae:5278.671447+3.005904	test-mae:5278.690953+23.874880 
    [23]	train-mae:5247.035794+2.970425	test-mae:5247.056292+23.772150 
    [24]	train-mae:5215.576013+2.958963	test-mae:5215.594264+23.654235 
    [25]	train-mae:5184.314405+2.952665	test-mae:5184.333211+23.533665 
    [26]	train-mae:5153.236882+2.930546	test-mae:5153.255159+23.434216 
    [27]	train-mae:5122.343978+2.911950	test-mae:5122.357982+23.329892 
    [28]	train-mae:5091.634339+2.892426	test-mae:5091.648343+23.205309 
    [29]	train-mae:5061.120789+2.875599	test-mae:5061.141443+23.095831 
    [30]	train-mae:5030.790431+2.856670	test-mae:5030.815901+22.979918 
    [31]	train-mae:5000.634966+2.848502	test-mae:5000.661037+22.868790 
    [32]	train-mae:4970.668630+2.834059	test-mae:4970.694211+22.761311 
    [33]	train-mae:4940.866344+2.811005	test-mae:4940.882253+22.659276 
    [34]	train-mae:4911.247739+2.803760	test-mae:4911.264843+22.535664 
    [35]	train-mae:4881.800631+2.796577	test-mae:4881.821766+22.425933 
    [36]	train-mae:4852.544665+2.788147	test-mae:4852.568378+22.309897 
    [37]	train-mae:4823.455696+2.778751	test-mae:4823.473691+22.199279 
    [38]	train-mae:4794.541049+2.757648	test-mae:4794.550608+22.091390 
    [39]	train-mae:4765.806290+2.736534	test-mae:4765.814577+21.983451 
    [40]	train-mae:4737.240991+2.719389	test-mae:4737.255468+21.880879 
    [41]	train-mae:4708.850550+2.707747	test-mae:4708.863619+21.782380 
    [42]	train-mae:4680.607897+2.691996	test-mae:4680.624446+21.691397 
    [43]	train-mae:4652.551018+2.677977	test-mae:4652.567109+21.586890 
    [44]	train-mae:4624.663957+2.663906	test-mae:4624.675765+21.489286 
    [45]	train-mae:4596.941276+2.652709	test-mae:4596.957366+21.393400 
    [46]	train-mae:4569.384721+2.622647	test-mae:4569.399799+21.309333 
    [47]	train-mae:4541.995785+2.604911	test-mae:4542.013228+21.212106 
    [48]	train-mae:4514.763267+2.598537	test-mae:4514.783888+21.113198 
    [49]	train-mae:4487.705715+2.587769	test-mae:4487.733600+21.021791 
    [50]	train-mae:4460.805317+2.568988	test-mae:4460.831791+20.931603 
    [51]	train-mae:4434.056862+2.554136	test-mae:4434.081229+20.843199 
    [52]	train-mae:4407.472532+2.528485	test-mae:4407.506917+20.771157 
    [53]	train-mae:4381.053926+2.511911	test-mae:4381.091811+20.681852 
    [54]	train-mae:4354.793281+2.488202	test-mae:4354.836131+20.597511 
    [55]	train-mae:4328.711617+2.480064	test-mae:4328.754719+20.503646 
    [56]	train-mae:4302.769237+2.463481	test-mae:4302.827174+20.416447 
    [57]	train-mae:4276.981120+2.448268	test-mae:4277.051318+20.325875 
    [58]	train-mae:4251.341763+2.436811	test-mae:4251.419428+20.233566 
    [59]	train-mae:4225.855796+2.418744	test-mae:4225.935711+20.157767 
    [60]	train-mae:4200.527514+2.415951	test-mae:4200.606944+20.051743 
    [61]	train-mae:4175.355642+2.405517	test-mae:4175.445126+19.970431 
    [62]	train-mae:4150.328802+2.384640	test-mae:4150.417482+19.890838 
    [63]	train-mae:4125.452528+2.379164	test-mae:4125.543573+19.800618 
    [64]	train-mae:4100.734937+2.369973	test-mae:4100.826945+19.718025 
    [65]	train-mae:4076.152337+2.356433	test-mae:4076.246188+19.643617 
    [66]	train-mae:4051.719146+2.352774	test-mae:4051.799477+19.553227 
    [67]	train-mae:4027.456791+2.337729	test-mae:4027.544100+19.481568 
    [68]	train-mae:4003.333817+2.335526	test-mae:4003.425839+19.399274 
    [69]	train-mae:3979.351005+2.329468	test-mae:3979.444054+19.328023 
    [70]	train-mae:3955.507805+2.310366	test-mae:3955.588525+19.256387 
    [71]	train-mae:3931.805552+2.292437	test-mae:3931.892732+19.193261 
    [72]	train-mae:3908.252963+2.285967	test-mae:3908.340215+19.132603 
    [73]	train-mae:3884.839455+2.274195	test-mae:3884.928278+19.067988 
    [74]	train-mae:3861.562091+2.253592	test-mae:3861.649184+19.008471 
    [75]	train-mae:3838.426638+2.251828	test-mae:3838.521147+18.935785 
    [76]	train-mae:3815.438808+2.227179	test-mae:3815.542468+18.888939 
    [77]	train-mae:3792.601926+2.215374	test-mae:3792.701129+18.830721 
    [78]	train-mae:3769.892594+2.204992	test-mae:3769.995054+18.767716 
    [79]	train-mae:3747.307670+2.199614	test-mae:3747.417237+18.692071 
    [80]	train-mae:3724.867123+2.185698	test-mae:3724.980009+18.634705 
    [81]	train-mae:3702.555906+2.178495	test-mae:3702.670709+18.568548 
    [82]	train-mae:3680.385968+2.161345	test-mae:3680.506454+18.522352 
    [83]	train-mae:3658.351228+2.151542	test-mae:3658.474439+18.465647 
    [84]	train-mae:3636.461730+2.144470	test-mae:3636.590957+18.401964 
    [85]	train-mae:3614.682121+2.141143	test-mae:3614.817269+18.337599 
    [86]	train-mae:3593.050523+2.138831	test-mae:3593.186610+18.280527 
    [87]	train-mae:3571.539732+2.121421	test-mae:3571.674045+18.234834 
    [88]	train-mae:3550.162135+2.114052	test-mae:3550.293748+18.176263 
    [89]	train-mae:3528.904800+2.101698	test-mae:3529.044330+18.115789 
    [90]	train-mae:3507.777798+2.094940	test-mae:3507.917675+18.068817 
    [91]	train-mae:3486.774824+2.093333	test-mae:3486.903310+18.006045 
    [92]	train-mae:3465.912963+2.094727	test-mae:3466.042809+17.951177 
    [93]	train-mae:3445.166292+2.084981	test-mae:3445.290250+17.902870 
    [94]	train-mae:3424.550273+2.081616	test-mae:3424.686238+17.846686 
    [95]	train-mae:3404.056769+2.087719	test-mae:3404.200904+17.777688 
    [96]	train-mae:3383.690504+2.075899	test-mae:3383.836386+17.722626 
    [97]	train-mae:3363.446084+2.057319	test-mae:3363.594497+17.681745 
    [98]	train-mae:3343.325970+2.036424	test-mae:3343.484317+17.640184 
    [99]	train-mae:3323.325803+2.021206	test-mae:3323.491517+17.596377 
    [100]	train-mae:3303.448547+2.006646	test-mae:3303.611227+17.547551 
    [101]	train-mae:3283.696844+1.992556	test-mae:3283.871831+17.496580 
    [102]	train-mae:3264.070668+1.968811	test-mae:3264.239923+17.460133 
    [103]	train-mae:3244.558464+1.951579	test-mae:3244.731322+17.423730 
    [104]	train-mae:3225.166799+1.939196	test-mae:3225.345866+17.368408 
    [105]	train-mae:3205.882236+1.917978	test-mae:3206.071430+17.321960 
    [106]	train-mae:3186.716387+1.906498	test-mae:3186.918567+17.267638 
    [107]	train-mae:3167.665848+1.885439	test-mae:3167.883422+17.225278 
    [108]	train-mae:3148.728680+1.880168	test-mae:3148.954752+17.164553 
    [109]	train-mae:3129.922856+1.873427	test-mae:3130.154826+17.116788 
    [110]	train-mae:3111.233206+1.873052	test-mae:3111.468412+17.053902 
    [111]	train-mae:3092.649117+1.867339	test-mae:3092.882366+17.000293 
    [112]	train-mae:3074.174179+1.840659	test-mae:3074.410016+16.960093 
    [113]	train-mae:3055.810776+1.823598	test-mae:3056.051129+16.914985 
    [114]	train-mae:3037.543698+1.815826	test-mae:3037.785577+16.860912 
    [115]	train-mae:3019.400309+1.803664	test-mae:3019.645035+16.819528 
    [116]	train-mae:3001.373811+1.792127	test-mae:3001.631973+16.770088 
    [117]	train-mae:2983.436696+1.784233	test-mae:2983.693485+16.720140 
    [118]	train-mae:2965.620100+1.785351	test-mae:2965.894636+16.654188 
    [119]	train-mae:2947.906212+1.780437	test-mae:2948.195559+16.594618 
    [120]	train-mae:2930.304487+1.771970	test-mae:2930.596426+16.542659 
    [121]	train-mae:2912.808275+1.766262	test-mae:2913.104167+16.485910 
    [122]	train-mae:2895.416020+1.749593	test-mae:2895.731068+16.451144 
    [123]	train-mae:2878.135354+1.734547	test-mae:2878.465056+16.413692 
    [124]	train-mae:2860.967064+1.724889	test-mae:2861.303188+16.372040 
    [125]	train-mae:2843.898660+1.713363	test-mae:2844.240159+16.321467 
    [126]	train-mae:2826.927407+1.704171	test-mae:2827.273663+16.285632 
    [127]	train-mae:2810.054403+1.704646	test-mae:2810.408718+16.238800 
    [128]	train-mae:2793.279071+1.702231	test-mae:2793.639725+16.186678 
    [129]	train-mae:2776.611837+1.699701	test-mae:2776.973889+16.132790 
    [130]	train-mae:2760.051686+1.679079	test-mae:2760.420143+16.102611 
    [131]	train-mae:2743.585376+1.658633	test-mae:2743.950205+16.074119 
    [132]	train-mae:2727.222355+1.646232	test-mae:2727.588360+16.047531 
    [133]	train-mae:2710.958433+1.629511	test-mae:2711.328590+16.014708 
    [134]	train-mae:2694.790159+1.608352	test-mae:2695.161411+15.984140 
    [135]	train-mae:2678.717582+1.594990	test-mae:2679.092751+15.942225 
    [136]	train-mae:2662.746138+1.580970	test-mae:2663.128653+15.911883 
    [137]	train-mae:2646.874368+1.579668	test-mae:2647.260829+15.862408 
    [138]	train-mae:2631.091670+1.577780	test-mae:2631.476349+15.816567 
    [139]	train-mae:2615.398619+1.569340	test-mae:2615.781238+15.783741 
    [140]	train-mae:2599.825156+1.563506	test-mae:2600.210654+15.744642 
    [141]	train-mae:2584.336917+1.553336	test-mae:2584.723159+15.707084 
    [142]	train-mae:2568.934774+1.538297	test-mae:2569.320744+15.681143 
    [143]	train-mae:2553.633630+1.526885	test-mae:2554.014400+15.654594 
    [144]	train-mae:2538.407862+1.507582	test-mae:2538.793488+15.625932 
    [145]	train-mae:2523.280511+1.500103	test-mae:2523.663467+15.601934 
    [146]	train-mae:2508.246004+1.489101	test-mae:2508.642038+15.568082 
    [147]	train-mae:2493.304707+1.485805	test-mae:2493.702534+15.530213 
    [148]	train-mae:2478.451228+1.474779	test-mae:2478.852452+15.496023 
    [149]	train-mae:2463.683230+1.468867	test-mae:2464.091353+15.458802 
    [150]	train-mae:2449.010250+1.455493	test-mae:2449.422770+15.421399 
    [151]	train-mae:2434.419930+1.445056	test-mae:2434.839873+15.398017 
    [152]	train-mae:2419.919708+1.439688	test-mae:2420.354569+15.372231 
    [153]	train-mae:2405.509526+1.432103	test-mae:2405.944833+15.341587 
    [154]	train-mae:2391.194512+1.426582	test-mae:2391.630485+15.303639 
    [155]	train-mae:2376.951219+1.423593	test-mae:2377.398165+15.275069 
    [156]	train-mae:2362.801131+1.416698	test-mae:2363.251896+15.253758 
    [157]	train-mae:2348.742901+1.415653	test-mae:2349.218656+15.226382 
    [158]	train-mae:2334.756541+1.406759	test-mae:2335.237107+15.195737 
    [159]	train-mae:2320.860429+1.396333	test-mae:2321.357676+15.188170 
    [160]	train-mae:2307.056451+1.373424	test-mae:2307.558881+15.176982 
    [161]	train-mae:2293.331289+1.362871	test-mae:2293.843072+15.158760 
    [162]	train-mae:2279.690826+1.361122	test-mae:2280.198139+15.126060 
    [163]	train-mae:2266.127660+1.355677	test-mae:2266.644129+15.090413 
    [164]	train-mae:2252.652603+1.347667	test-mae:2253.176286+15.069761 
    [165]	train-mae:2239.264990+1.340814	test-mae:2239.800066+15.046532 
    [166]	train-mae:2225.945970+1.324230	test-mae:2226.492137+15.038074 
    [167]	train-mae:2212.724887+1.325802	test-mae:2213.273532+15.012297 
    [168]	train-mae:2199.576150+1.319415	test-mae:2200.131914+14.989720 
    [169]	train-mae:2186.500389+1.315603	test-mae:2187.067917+14.958311 
    [170]	train-mae:2173.505804+1.306635	test-mae:2174.082381+14.944056 
    [171]	train-mae:2160.589550+1.300864	test-mae:2161.180279+14.921502 
    [172]	train-mae:2147.748576+1.306602	test-mae:2148.351821+14.886176 
    [173]	train-mae:2134.987278+1.290717	test-mae:2135.605769+14.871572 
    [174]	train-mae:2122.309934+1.280960	test-mae:2122.940310+14.849981 
    [175]	train-mae:2109.708065+1.289749	test-mae:2110.350410+14.838519 
    [176]	train-mae:2097.186331+1.286300	test-mae:2097.839550+14.820114 
    [177]	train-mae:2084.744485+1.282599	test-mae:2085.408163+14.809408 
    [178]	train-mae:2072.381569+1.279259	test-mae:2073.059772+14.780235 
    [179]	train-mae:2060.082852+1.277766	test-mae:2060.770901+14.736242 
    [180]	train-mae:2047.867643+1.273972	test-mae:2048.564009+14.707351 
    [181]	train-mae:2035.718692+1.271767	test-mae:2036.426612+14.680989 
    [182]	train-mae:2023.656034+1.268339	test-mae:2024.370323+14.668304 
    [183]	train-mae:2011.646491+1.259773	test-mae:2012.385308+14.655952 
    [184]	train-mae:1999.722933+1.259421	test-mae:2000.475347+14.631588 
    [185]	train-mae:1987.862780+1.247489	test-mae:1988.627892+14.609315 
    [186]	train-mae:1976.087496+1.237075	test-mae:1976.863934+14.592758 
    [187]	train-mae:1964.383143+1.241333	test-mae:1965.178446+14.566060 
    [188]	train-mae:1952.747571+1.234915	test-mae:1953.553305+14.541052 
    [189]	train-mae:1941.189762+1.222627	test-mae:1942.006709+14.511642 
    [190]	train-mae:1929.708002+1.220785	test-mae:1930.541711+14.481047 
    [191]	train-mae:1918.283293+1.214211	test-mae:1919.138538+14.463584 
    [192]	train-mae:1906.934119+1.200884	test-mae:1907.804944+14.431166 
    [193]	train-mae:1895.662853+1.194648	test-mae:1896.551082+14.405003 
    [194]	train-mae:1884.447314+1.186060	test-mae:1885.353683+14.384137 
    [195]	train-mae:1873.300463+1.181466	test-mae:1874.227295+14.353250 
    [196]	train-mae:1862.228063+1.182840	test-mae:1863.167197+14.315465 
    [197]	train-mae:1851.234605+1.170998	test-mae:1852.187996+14.295420 
    [198]	train-mae:1840.307712+1.163972	test-mae:1841.279534+14.268266 
    [199]	train-mae:1829.437730+1.149893	test-mae:1830.416750+14.251273 
    [200]	train-mae:1818.642962+1.147746	test-mae:1819.635690+14.220252 
    [201]	train-mae:1807.918901+1.136197	test-mae:1808.924768+14.199450 
    [202]	train-mae:1797.247923+1.132997	test-mae:1798.270023+14.168250 
    [203]	train-mae:1786.647354+1.138407	test-mae:1787.691975+14.139407 
    [204]	train-mae:1776.118615+1.140152	test-mae:1777.176813+14.109897 
    [205]	train-mae:1765.656730+1.137834	test-mae:1766.725052+14.086291 
    [206]	train-mae:1755.258724+1.138031	test-mae:1756.328470+14.052131 
    [207]	train-mae:1744.920212+1.136837	test-mae:1745.999330+14.026890 
    [208]	train-mae:1734.649570+1.141834	test-mae:1735.741247+14.010992 
    [209]	train-mae:1724.433791+1.138754	test-mae:1725.543771+13.993022 
    [210]	train-mae:1714.294603+1.141455	test-mae:1715.419276+13.959638 
    [211]	train-mae:1704.217039+1.137200	test-mae:1705.351306+13.930260 
    [212]	train-mae:1694.192046+1.138913	test-mae:1695.349535+13.907377 
    [213]	train-mae:1684.224255+1.138394	test-mae:1685.392001+13.870802 
    [214]	train-mae:1674.321789+1.132978	test-mae:1675.511117+13.845903 
    [215]	train-mae:1664.484362+1.126160	test-mae:1665.679468+13.829998 
    [216]	train-mae:1654.706138+1.124778	test-mae:1655.922756+13.797508 
    [217]	train-mae:1644.977923+1.121362	test-mae:1646.202264+13.768436 
    [218]	train-mae:1635.319380+1.103305	test-mae:1636.562836+13.738732 
    [219]	train-mae:1625.722913+1.108679	test-mae:1626.988398+13.693592 
    [220]	train-mae:1616.189590+1.087356	test-mae:1617.475053+13.671662 
    [221]	train-mae:1606.716108+1.085098	test-mae:1608.022890+13.636201 
    [222]	train-mae:1597.304488+1.074868	test-mae:1598.630682+13.602139 
    [223]	train-mae:1587.936881+1.081070	test-mae:1589.279563+13.570395 
    [224]	train-mae:1578.630541+1.083038	test-mae:1579.989228+13.532123 
    [225]	train-mae:1569.385774+1.084934	test-mae:1570.764872+13.497073 
    [226]	train-mae:1560.200872+1.081582	test-mae:1561.598722+13.463268 
    [227]	train-mae:1551.069656+1.068243	test-mae:1552.488306+13.438935 
    [228]	train-mae:1542.006988+1.062325	test-mae:1543.455755+13.405223 
    [229]	train-mae:1532.985766+1.066418	test-mae:1534.454550+13.357923 
    [230]	train-mae:1524.034818+1.064214	test-mae:1525.533989+13.320963 
    [231]	train-mae:1515.135147+1.065050	test-mae:1516.646341+13.286150 
    [232]	train-mae:1506.280415+1.063075	test-mae:1507.814976+13.242113 
    [233]	train-mae:1497.495090+1.055770	test-mae:1499.049971+13.215524 
    [234]	train-mae:1488.760348+1.048330	test-mae:1490.343265+13.192855 
    [235]	train-mae:1480.071686+1.033235	test-mae:1481.689458+13.152956 
    [236]	train-mae:1471.442948+1.034406	test-mae:1473.084538+13.106225 
    [237]	train-mae:1462.872131+1.032370	test-mae:1464.536540+13.064660 
    [238]	train-mae:1454.351782+1.031832	test-mae:1456.035979+13.018747 
    [239]	train-mae:1445.883378+1.019364	test-mae:1447.589385+12.990556 
    [240]	train-mae:1437.464883+1.011836	test-mae:1439.191472+12.948340 
    [241]	train-mae:1429.106780+1.015740	test-mae:1430.864634+12.907791 
    [242]	train-mae:1420.802869+1.019562	test-mae:1422.588313+12.869433 
    [243]	train-mae:1412.556965+1.007227	test-mae:1414.363262+12.848394 
    [244]	train-mae:1404.356611+1.001187	test-mae:1406.187537+12.813119 
    [245]	train-mae:1396.205491+1.000243	test-mae:1398.067604+12.781172 
    [246]	train-mae:1388.112184+0.991734	test-mae:1389.994159+12.757496 
    [247]	train-mae:1380.059314+0.985027	test-mae:1381.958638+12.730375 
    [248]	train-mae:1372.054638+0.982460	test-mae:1373.979542+12.697767 
    [249]	train-mae:1364.117566+0.991474	test-mae:1366.064203+12.652777 
    [250]	train-mae:1356.230627+0.993924	test-mae:1358.199337+12.611676 
    [251]	train-mae:1348.390948+0.991523	test-mae:1350.393618+12.585740 
    [252]	train-mae:1340.599276+0.986246	test-mae:1342.626878+12.550645 
    [253]	train-mae:1332.862142+0.985283	test-mae:1334.920985+12.512888 
    [254]	train-mae:1325.167869+0.978731	test-mae:1327.252628+12.467737 
    [255]	train-mae:1317.528636+0.974480	test-mae:1319.641540+12.427868 
    [256]	train-mae:1309.931494+0.971004	test-mae:1312.084860+12.375530 
    [257]	train-mae:1302.392783+0.958528	test-mae:1304.587187+12.330358 
    [258]	train-mae:1294.912035+0.956519	test-mae:1297.139420+12.277920 
    [259]	train-mae:1287.479987+0.948683	test-mae:1289.742611+12.220510 
    [260]	train-mae:1280.097048+0.942899	test-mae:1282.390340+12.174698 
    [261]	train-mae:1272.766584+0.956171	test-mae:1275.094672+12.117741 
    [262]	train-mae:1265.481331+0.954004	test-mae:1267.841642+12.067243 
    [263]	train-mae:1258.252894+0.957608	test-mae:1260.647461+12.030151 
    [264]	train-mae:1251.079430+0.958603	test-mae:1253.505517+11.994518 
    [265]	train-mae:1243.942784+0.958635	test-mae:1246.402569+11.940394 
    [266]	train-mae:1236.855588+0.961588	test-mae:1239.341646+11.895759 
    [267]	train-mae:1229.819385+0.958101	test-mae:1232.342563+11.844614 
    [268]	train-mae:1222.823904+0.961865	test-mae:1225.375232+11.796068 
    [269]	train-mae:1215.877692+0.954655	test-mae:1218.462829+11.746865 
    [270]	train-mae:1208.984629+0.949199	test-mae:1211.601767+11.716372 
    [271]	train-mae:1202.134710+0.928837	test-mae:1204.788363+11.701733 
    [272]	train-mae:1195.347281+0.931496	test-mae:1198.022800+11.658502 
    [273]	train-mae:1188.603523+0.925312	test-mae:1191.312753+11.631472 
    [274]	train-mae:1181.901683+0.915142	test-mae:1184.647178+11.599877 
    [275]	train-mae:1175.235834+0.914132	test-mae:1178.008901+11.560781 
    [276]	train-mae:1168.629431+0.916357	test-mae:1171.419508+11.528323 
    [277]	train-mae:1162.070286+0.914735	test-mae:1164.905866+11.485285 
    [278]	train-mae:1155.545255+0.906255	test-mae:1158.409871+11.452829 
    [279]	train-mae:1149.081158+0.907006	test-mae:1151.979735+11.415203 
    [280]	train-mae:1142.652529+0.908131	test-mae:1145.592617+11.376787 
    [281]	train-mae:1136.261796+0.907499	test-mae:1139.230400+11.340143 
    [282]	train-mae:1129.924804+0.902963	test-mae:1132.917219+11.306400 
    [283]	train-mae:1123.628991+0.904871	test-mae:1126.653313+11.252703 
    [284]	train-mae:1117.378490+0.904540	test-mae:1120.436111+11.201410 
    [285]	train-mae:1111.171419+0.900827	test-mae:1114.267206+11.171403 
    [286]	train-mae:1105.002328+0.908159	test-mae:1108.137916+11.128264 
    [287]	train-mae:1098.878071+0.913378	test-mae:1102.043246+11.098913 
    [288]	train-mae:1092.798731+0.915003	test-mae:1096.002446+11.069254 
    [289]	train-mae:1086.771841+0.915040	test-mae:1090.008822+11.033193 
    [290]	train-mae:1080.779519+0.923851	test-mae:1084.048000+11.003765 
    [291]	train-mae:1074.828732+0.931148	test-mae:1078.129831+10.971682 
    [292]	train-mae:1068.906321+0.928119	test-mae:1072.240399+10.946537 
    [293]	train-mae:1063.040570+0.918948	test-mae:1066.408826+10.918472 
    [294]	train-mae:1057.207848+0.919006	test-mae:1060.613410+10.879564 
    [295]	train-mae:1051.412165+0.926188	test-mae:1054.861846+10.828276 
    [296]	train-mae:1045.657794+0.933728	test-mae:1049.149223+10.777415 
    [297]	train-mae:1039.944065+0.929643	test-mae:1043.482453+10.745297 
    [298]	train-mae:1034.266426+0.925351	test-mae:1037.846200+10.722825 
    [299]	train-mae:1028.627125+0.914738	test-mae:1032.240766+10.692854 
    [300]	train-mae:1023.036895+0.918061	test-mae:1026.689766+10.664815 
    [301]	train-mae:1017.483918+0.928690	test-mae:1021.172158+10.643900 
    [302]	train-mae:1011.956259+0.933959	test-mae:1015.686894+10.598571 
    [303]	train-mae:1006.486528+0.935179	test-mae:1010.260080+10.584445 
    [304]	train-mae:1001.045364+0.936618	test-mae:1004.865172+10.570953 
    [305]	train-mae:995.641153+0.940368	test-mae:999.499375+10.540546 
    [306]	train-mae:990.274261+0.937695	test-mae:994.162479+10.531076 
    [307]	train-mae:984.937837+0.941297	test-mae:988.868713+10.509901 
    [308]	train-mae:979.642045+0.942630	test-mae:983.617039+10.506711 
    [309]	train-mae:974.381188+0.951222	test-mae:978.397984+10.484839 
    [310]	train-mae:969.156503+0.968905	test-mae:973.209635+10.449916 
    [311]	train-mae:963.983446+0.988230	test-mae:968.072789+10.426508 
    [312]	train-mae:958.833489+0.991218	test-mae:962.963513+10.413466 
    [313]	train-mae:953.721860+0.991966	test-mae:957.892881+10.407839 
    [314]	train-mae:948.653480+0.994600	test-mae:952.866247+10.392987 
    [315]	train-mae:943.623223+0.995635	test-mae:947.870353+10.385257 
    [316]	train-mae:938.618919+0.991102	test-mae:942.906279+10.364993 
    [317]	train-mae:933.654243+0.992666	test-mae:937.970814+10.357841 
    [318]	train-mae:928.726767+1.004579	test-mae:933.073903+10.342046 
    [319]	train-mae:923.837293+1.011659	test-mae:928.211318+10.335377 
    [320]	train-mae:918.975129+1.010925	test-mae:923.388603+10.314489 
    [321]	train-mae:914.150127+1.012934	test-mae:918.593476+10.306117 
    [322]	train-mae:909.354624+1.018852	test-mae:913.843111+10.281787 
    [323]	train-mae:904.607432+1.021692	test-mae:909.134505+10.279171 
    [324]	train-mae:899.886498+1.032228	test-mae:904.452588+10.266882 
    [325]	train-mae:895.197691+1.040461	test-mae:899.812752+10.254856 
    [326]	train-mae:890.546142+1.042095	test-mae:895.207750+10.237288 
    [327]	train-mae:885.926147+1.043454	test-mae:890.615129+10.235103 
    [328]	train-mae:881.341361+1.042948	test-mae:886.059679+10.227712 
    [329]	train-mae:876.782303+1.045576	test-mae:881.538676+10.210053 
    [330]	train-mae:872.258491+1.057003	test-mae:877.044601+10.183127 
    [331]	train-mae:867.764469+1.057798	test-mae:872.574135+10.178426 
    [332]	train-mae:863.302484+1.062565	test-mae:868.156757+10.172716 
    [333]	train-mae:858.871857+1.067159	test-mae:863.769550+10.159755 
    [334]	train-mae:854.465551+1.080597	test-mae:859.410382+10.146423 
    [335]	train-mae:850.094971+1.078280	test-mae:855.084650+10.139598 
    [336]	train-mae:845.757030+1.081874	test-mae:850.788669+10.124477 
    [337]	train-mae:841.456159+1.084291	test-mae:846.526586+10.110803 
    [338]	train-mae:837.177903+1.080769	test-mae:842.284383+10.096926 
    [339]	train-mae:832.928265+1.075869	test-mae:838.064507+10.085737 
    [340]	train-mae:828.721697+1.074708	test-mae:833.894528+10.077724 
    [341]	train-mae:824.534966+1.076483	test-mae:829.746698+10.065128 
    [342]	train-mae:820.377427+1.067302	test-mae:825.610476+10.067227 
    [343]	train-mae:816.252611+1.072447	test-mae:821.526348+10.054171 
    [344]	train-mae:812.152217+1.069782	test-mae:817.460759+10.041785 
    [345]	train-mae:808.079439+1.071290	test-mae:813.420735+10.027590 
    [346]	train-mae:804.040011+1.074147	test-mae:809.420669+10.012370 
    [347]	train-mae:800.027102+1.075344	test-mae:805.440770+9.996852 
    [348]	train-mae:796.043752+1.076193	test-mae:801.493260+9.974927 
    [349]	train-mae:792.094593+1.072392	test-mae:797.581803+9.962814 
    [350]	train-mae:788.165506+1.069184	test-mae:793.690477+9.955683 
    [351]	train-mae:784.271691+1.064317	test-mae:789.824357+9.952828 
    [352]	train-mae:780.385217+1.068214	test-mae:785.971918+9.942667 
    [353]	train-mae:776.534597+1.068425	test-mae:782.154701+9.931208 
    [354]	train-mae:772.717591+1.065642	test-mae:778.380431+9.919930 
    [355]	train-mae:768.929989+1.064673	test-mae:774.632114+9.904204 
    [356]	train-mae:765.164608+1.063291	test-mae:770.903852+9.888740 
    [357]	train-mae:761.433681+1.062617	test-mae:767.208245+9.893516 
    [358]	train-mae:757.720303+1.059469	test-mae:763.525570+9.878217 
    [359]	train-mae:754.052187+1.058765	test-mae:759.892451+9.863265 
    [360]	train-mae:750.411297+1.059444	test-mae:756.280662+9.855747 
    [361]	train-mae:746.793756+1.046557	test-mae:752.701747+9.841489 
    [362]	train-mae:743.200341+1.049858	test-mae:749.150155+9.816691 
    [363]	train-mae:739.632371+1.058249	test-mae:745.631904+9.793323 
    [364]	train-mae:736.088069+1.057345	test-mae:742.122404+9.785878 
    [365]	train-mae:732.561058+1.048205	test-mae:738.628002+9.788025 
    [366]	train-mae:729.076243+1.048044	test-mae:735.179818+9.772984 
    [367]	train-mae:725.607855+1.042075	test-mae:731.749352+9.778832 
    [368]	train-mae:722.166013+1.036954	test-mae:728.345591+9.781078 
    [369]	train-mae:718.753052+1.038548	test-mae:724.957980+9.767931 
    [370]	train-mae:715.362821+1.042433	test-mae:721.606713+9.764225 
    [371]	train-mae:711.999386+1.039046	test-mae:718.278930+9.761135 
    [372]	train-mae:708.653371+1.036271	test-mae:714.960346+9.760172 
    [373]	train-mae:705.341049+1.028675	test-mae:711.688572+9.761481 
    [374]	train-mae:702.054906+1.021496	test-mae:708.435658+9.753532 
    [375]	train-mae:698.792202+1.020581	test-mae:705.204893+9.748062 
    [376]	train-mae:695.551333+1.018500	test-mae:701.998620+9.730742 
    [377]	train-mae:692.343391+1.021638	test-mae:698.821421+9.715807 
    [378]	train-mae:689.160229+1.023116	test-mae:695.679533+9.705083 
    [379]	train-mae:685.980910+1.014692	test-mae:692.526948+9.692997 
    [380]	train-mae:682.838846+1.018584	test-mae:689.418036+9.670208 
    [381]	train-mae:679.714851+1.009559	test-mae:686.336031+9.663007 
    [382]	train-mae:676.624944+1.011467	test-mae:683.288583+9.651548 
    [383]	train-mae:673.555926+1.014616	test-mae:680.246754+9.647742 
    [384]	train-mae:670.506478+1.023630	test-mae:677.229670+9.621541 
    [385]	train-mae:667.473591+1.024625	test-mae:674.228295+9.609387 
    [386]	train-mae:664.464911+1.021034	test-mae:671.247550+9.604589 
    [387]	train-mae:661.481425+1.028211	test-mae:668.303853+9.591076 
    [388]	train-mae:658.509394+1.025098	test-mae:665.370051+9.583794 
    [389]	train-mae:655.569738+1.020012	test-mae:662.462101+9.571573 
    [390]	train-mae:652.647375+1.016359	test-mae:659.571752+9.571660 
    [391]	train-mae:649.745406+1.011869	test-mae:656.708473+9.555454 
    [392]	train-mae:646.866663+1.008675	test-mae:653.869168+9.546677 
    [393]	train-mae:644.012304+1.005569	test-mae:651.047673+9.530020 
    [394]	train-mae:641.176677+1.002442	test-mae:648.255501+9.528078 
    [395]	train-mae:638.378157+1.007083	test-mae:645.493511+9.505050 
    [396]	train-mae:635.574702+0.998925	test-mae:642.725796+9.502346 
    [397]	train-mae:632.804774+0.994925	test-mae:639.988940+9.491418 
    [398]	train-mae:630.056266+0.993366	test-mae:637.273082+9.461993 
    [399]	train-mae:627.322092+0.996696	test-mae:634.578352+9.451099 
    [400]	train-mae:624.611689+0.994992	test-mae:631.912612+9.419632 
    [401]	train-mae:621.920172+0.980005	test-mae:629.252368+9.406665 
    [402]	train-mae:619.250357+0.965826	test-mae:626.621028+9.383486 
    [403]	train-mae:616.599170+0.960063	test-mae:624.005922+9.372041 
    [404]	train-mae:613.972499+0.950769	test-mae:621.407331+9.357820 
    [405]	train-mae:611.379009+0.949875	test-mae:618.847205+9.351465 
    [406]	train-mae:608.802204+0.943750	test-mae:616.307837+9.331820 
    [407]	train-mae:606.241251+0.945347	test-mae:613.780480+9.308554 
    [408]	train-mae:603.697200+0.944253	test-mae:611.271734+9.288728 
    [409]	train-mae:601.174663+0.948768	test-mae:608.787268+9.260830 
    [410]	train-mae:598.666879+0.945402	test-mae:606.319008+9.233340 
    [411]	train-mae:596.179843+0.946196	test-mae:603.866038+9.212077 
    [412]	train-mae:593.710175+0.939132	test-mae:601.435498+9.206318 
    [413]	train-mae:591.266177+0.935703	test-mae:599.031532+9.194983 
    [414]	train-mae:588.842633+0.935110	test-mae:596.652525+9.176996 
    [415]	train-mae:586.423262+0.942958	test-mae:594.272306+9.150094 
    [416]	train-mae:584.025333+0.940646	test-mae:591.905032+9.134499 
    [417]	train-mae:581.660895+0.942270	test-mae:589.575199+9.136069 
    [418]	train-mae:579.310193+0.935438	test-mae:587.267733+9.133119 
    [419]	train-mae:576.975315+0.930864	test-mae:584.979097+9.135405 
    [420]	train-mae:574.654756+0.930508	test-mae:582.684661+9.130561 
    [421]	train-mae:572.359233+0.926239	test-mae:580.424067+9.136856 
    [422]	train-mae:570.085449+0.928435	test-mae:578.186932+9.129714 
    [423]	train-mae:567.828795+0.927966	test-mae:575.955641+9.127392 
    [424]	train-mae:565.587674+0.924456	test-mae:573.746236+9.134643 
    [425]	train-mae:563.353493+0.924753	test-mae:571.542038+9.127521 
    [426]	train-mae:561.145394+0.914692	test-mae:569.374233+9.133725 
    [427]	train-mae:558.948086+0.908526	test-mae:567.215769+9.132156 
    [428]	train-mae:556.772405+0.909764	test-mae:565.076452+9.122680 
    [429]	train-mae:554.614769+0.904053	test-mae:562.960999+9.120853 
    [430]	train-mae:552.468391+0.901032	test-mae:560.846336+9.118006 
    [431]	train-mae:550.344076+0.894310	test-mae:558.756579+9.110540 
    [432]	train-mae:548.239898+0.891651	test-mae:556.689286+9.118022 
    [433]	train-mae:546.149817+0.885724	test-mae:554.638198+9.118648 
    [434]	train-mae:544.081986+0.888591	test-mae:552.602399+9.113364 
    [435]	train-mae:542.026475+0.881241	test-mae:550.582473+9.103254 
    [436]	train-mae:539.991437+0.884950	test-mae:548.581286+9.105422 
    [437]	train-mae:537.963878+0.882776	test-mae:546.588867+9.103031 
    [438]	train-mae:535.958767+0.880402	test-mae:544.624197+9.099135 
    [439]	train-mae:533.966412+0.877361	test-mae:542.668153+9.094236 
    [440]	train-mae:531.986028+0.865747	test-mae:540.716011+9.089246 
    [441]	train-mae:530.034182+0.869087	test-mae:538.802145+9.083904 
    [442]	train-mae:528.089457+0.867983	test-mae:536.895376+9.075914 
    [443]	train-mae:526.167520+0.862837	test-mae:535.011240+9.084539 
    [444]	train-mae:524.259753+0.855738	test-mae:533.141150+9.088609 
    [445]	train-mae:522.364707+0.848608	test-mae:531.274184+9.085695 
    [446]	train-mae:520.483680+0.852597	test-mae:529.426353+9.066131 
    [447]	train-mae:518.614154+0.852281	test-mae:527.585863+9.056132 
    [448]	train-mae:516.768958+0.844145	test-mae:525.769371+9.052650 
    [449]	train-mae:514.936064+0.842991	test-mae:523.972077+9.048844 
    [450]	train-mae:513.118572+0.837780	test-mae:522.182680+9.042349 
    [451]	train-mae:511.314552+0.841029	test-mae:520.402659+9.043008 
    [452]	train-mae:509.530470+0.839512	test-mae:518.649577+9.036242 
    [453]	train-mae:507.769131+0.845045	test-mae:516.931547+9.030453 
    [454]	train-mae:506.010547+0.842852	test-mae:515.206499+9.017378 
    [455]	train-mae:504.272182+0.838861	test-mae:513.495814+9.006491 
    [456]	train-mae:502.557112+0.843399	test-mae:511.809397+8.993512 
    [457]	train-mae:500.849393+0.848831	test-mae:510.136187+8.982942 
    [458]	train-mae:499.159670+0.845089	test-mae:508.475672+8.964931 
    [459]	train-mae:497.480429+0.838597	test-mae:506.829209+8.956629 
    [460]	train-mae:495.818294+0.839243	test-mae:505.196416+8.939982 
    [461]	train-mae:494.167437+0.840800	test-mae:503.582419+8.938159 
    [462]	train-mae:492.531505+0.841307	test-mae:501.981718+8.932490 
    [463]	train-mae:490.904673+0.834089	test-mae:500.395359+8.923479 
    [464]	train-mae:489.289482+0.833662	test-mae:498.818507+8.913097 
    [465]	train-mae:487.703528+0.836654	test-mae:497.264193+8.905558 
    [466]	train-mae:486.123410+0.831761	test-mae:495.720211+8.899706 
    [467]	train-mae:484.559360+0.826898	test-mae:494.192252+8.892141 
    [468]	train-mae:483.007898+0.826521	test-mae:492.671177+8.876035 
    [469]	train-mae:481.470365+0.825073	test-mae:491.166114+8.863034 
    [470]	train-mae:479.952266+0.821243	test-mae:489.676234+8.847492 
    [471]	train-mae:478.443956+0.818924	test-mae:488.196886+8.828462 
    [472]	train-mae:476.962229+0.821667	test-mae:486.734255+8.807394 
    [473]	train-mae:475.478405+0.810917	test-mae:485.272934+8.804920 
    [474]	train-mae:474.004011+0.803327	test-mae:483.819041+8.791832 
    [475]	train-mae:472.546769+0.805135	test-mae:482.390785+8.784750 
    [476]	train-mae:471.096993+0.802861	test-mae:480.977971+8.776262 
    [477]	train-mae:469.668875+0.805424	test-mae:479.575338+8.759897 
    [478]	train-mae:468.254802+0.802832	test-mae:478.195063+8.739901 
    [479]	train-mae:466.848591+0.805780	test-mae:476.823200+8.730193 
    [480]	train-mae:465.457996+0.806015	test-mae:475.464455+8.711659 
    [481]	train-mae:464.072319+0.810900	test-mae:474.108503+8.693846 
    [482]	train-mae:462.704662+0.807998	test-mae:472.769413+8.682073 
    [483]	train-mae:461.343603+0.806421	test-mae:471.437958+8.671063 
    [484]	train-mae:459.986797+0.798702	test-mae:470.110627+8.662903 
    [485]	train-mae:458.642968+0.800643	test-mae:468.800348+8.640301 
    [486]	train-mae:457.322606+0.805211	test-mae:467.512855+8.624939 
    [487]	train-mae:456.007858+0.803650	test-mae:466.224480+8.616356 
    [488]	train-mae:454.707008+0.804163	test-mae:464.956336+8.612634 
    [489]	train-mae:453.420972+0.798599	test-mae:463.699956+8.600976 
    [490]	train-mae:452.142834+0.793839	test-mae:462.450240+8.595365 
    [491]	train-mae:450.879161+0.791711	test-mae:461.221492+8.588390 
    [492]	train-mae:449.615919+0.792145	test-mae:459.997835+8.578439 
    [493]	train-mae:448.374527+0.789860	test-mae:458.796201+8.565714 
    [494]	train-mae:447.144156+0.791919	test-mae:457.586618+8.549735 
    [495]	train-mae:445.923649+0.791270	test-mae:456.398906+8.540097 
    [496]	train-mae:444.719630+0.792704	test-mae:455.227718+8.537209 
    [497]	train-mae:443.517032+0.793448	test-mae:454.062864+8.519444 
    [498]	train-mae:442.325505+0.795440	test-mae:452.904935+8.518216 
    [499]	train-mae:441.141374+0.801304	test-mae:451.756794+8.502403 
    [500]	train-mae:439.978054+0.800223	test-mae:450.629792+8.493654 
    [501]	train-mae:438.823755+0.800038	test-mae:449.502866+8.488849 
    [502]	train-mae:437.683457+0.800241	test-mae:448.398688+8.469673 
    [503]	train-mae:436.553875+0.806930	test-mae:447.300585+8.451947 
    [504]	train-mae:435.426840+0.800842	test-mae:446.209510+8.442787 
    [505]	train-mae:434.312296+0.805138	test-mae:445.132067+8.427581 
    [506]	train-mae:433.212621+0.806886	test-mae:444.062350+8.418001 
    [507]	train-mae:432.123657+0.807890	test-mae:442.998886+8.397557 
    [508]	train-mae:431.035396+0.815938	test-mae:441.943920+8.387581 
    [509]	train-mae:429.963650+0.821175	test-mae:440.900082+8.368388 
    [510]	train-mae:428.902839+0.826402	test-mae:439.874245+8.349975 
    [511]	train-mae:427.856181+0.825240	test-mae:438.867315+8.341244 
    [512]	train-mae:426.811447+0.820420	test-mae:437.859367+8.330993 
    [513]	train-mae:425.782857+0.823575	test-mae:436.864840+8.322053 
    [514]	train-mae:424.756838+0.827493	test-mae:435.872225+8.310422 
    [515]	train-mae:423.745027+0.832190	test-mae:434.893708+8.292082 
    [516]	train-mae:422.747750+0.831632	test-mae:433.923405+8.281715 
    [517]	train-mae:421.755909+0.833381	test-mae:432.966822+8.276187 
    [518]	train-mae:420.776715+0.832214	test-mae:432.021315+8.265470 
    [519]	train-mae:419.809666+0.834343	test-mae:431.084722+8.255014 
    [520]	train-mae:418.842612+0.833754	test-mae:430.160707+8.244184 
    [521]	train-mae:417.887608+0.831296	test-mae:429.240846+8.235974 
    [522]	train-mae:416.948267+0.824303	test-mae:428.335651+8.218478 
    [523]	train-mae:416.008964+0.822482	test-mae:427.434549+8.213302 
    [524]	train-mae:415.085056+0.816248	test-mae:426.540645+8.202432 
    [525]	train-mae:414.174199+0.813083	test-mae:425.669126+8.194045 
    [526]	train-mae:413.266095+0.808633	test-mae:424.788590+8.186929 
    [527]	train-mae:412.368309+0.808079	test-mae:423.923880+8.186683 
    [528]	train-mae:411.480375+0.805255	test-mae:423.067058+8.170450 
    [529]	train-mae:410.601322+0.804245	test-mae:422.225009+8.159398 
    [530]	train-mae:409.736246+0.802508	test-mae:421.390921+8.144350 
    [531]	train-mae:408.877143+0.796247	test-mae:420.569563+8.136769 
    [532]	train-mae:408.027449+0.801379	test-mae:419.751084+8.115206 
    [533]	train-mae:407.175209+0.810010	test-mae:418.936501+8.099403 
    [534]	train-mae:406.344335+0.817160	test-mae:418.137882+8.075407 
    [535]	train-mae:405.515160+0.819975	test-mae:417.339065+8.063223 
    [536]	train-mae:404.690140+0.823459	test-mae:416.551577+8.045571 
    [537]	train-mae:403.873974+0.823926	test-mae:415.776927+8.036599 
    [538]	train-mae:403.072902+0.819193	test-mae:415.010099+8.010351 
    [539]	train-mae:402.279013+0.824129	test-mae:414.245048+7.993082 
    [540]	train-mae:401.493808+0.820615	test-mae:413.496592+7.970916 
    [541]	train-mae:400.705803+0.821892	test-mae:412.752280+7.957740 
    [542]	train-mae:399.934164+0.815383	test-mae:412.022449+7.946543 
    [543]	train-mae:399.172935+0.814847	test-mae:411.298040+7.938348 
    [544]	train-mae:398.419287+0.816274	test-mae:410.576796+7.929549 
    [545]	train-mae:397.669402+0.808451	test-mae:409.862104+7.923412 
    [546]	train-mae:396.930796+0.806196	test-mae:409.158396+7.917644 
    [547]	train-mae:396.199906+0.798998	test-mae:408.462351+7.911425 
    [548]	train-mae:395.474294+0.795294	test-mae:407.770916+7.903837 
    [549]	train-mae:394.759389+0.793131	test-mae:407.092259+7.894640 
    [550]	train-mae:394.051482+0.795849	test-mae:406.414332+7.886001 
    [551]	train-mae:393.351471+0.802628	test-mae:405.745345+7.887374 
    [552]	train-mae:392.657142+0.809879	test-mae:405.084490+7.882965 
    [553]	train-mae:391.973815+0.807908	test-mae:404.431255+7.872484 
    [554]	train-mae:391.294117+0.812371	test-mae:403.791437+7.852938 
    [555]	train-mae:390.625881+0.816981	test-mae:403.145313+7.836098 
    [556]	train-mae:389.960654+0.817310	test-mae:402.508777+7.835560 
    [557]	train-mae:389.302094+0.815183	test-mae:401.880289+7.826127 
    [558]	train-mae:388.652810+0.808806	test-mae:401.261815+7.833274 
    [559]	train-mae:388.013190+0.818273	test-mae:400.655759+7.818493 
    [560]	train-mae:387.378532+0.821779	test-mae:400.056271+7.818001 
    [561]	train-mae:386.748077+0.823718	test-mae:399.455360+7.819529 
    [562]	train-mae:386.124047+0.823934	test-mae:398.866947+7.819518 
    [563]	train-mae:385.502575+0.825606	test-mae:398.275440+7.813618 
    [564]	train-mae:384.895012+0.832204	test-mae:397.695653+7.814899 
    [565]	train-mae:384.290834+0.832140	test-mae:397.120696+7.803728 
    [566]	train-mae:383.691547+0.825348	test-mae:396.546721+7.810848 
    [567]	train-mae:383.101885+0.822972	test-mae:395.982962+7.813807 
    [568]	train-mae:382.516046+0.821566	test-mae:395.429799+7.805318 
    [569]	train-mae:381.936021+0.822776	test-mae:394.881724+7.797468 
    [570]	train-mae:381.360243+0.828697	test-mae:394.338074+7.791043 
    [571]	train-mae:380.790650+0.828506	test-mae:393.793751+7.790056 
    [572]	train-mae:380.238944+0.823470	test-mae:393.276863+7.786691 
    [573]	train-mae:379.683629+0.819621	test-mae:392.758288+7.778320 
    [574]	train-mae:379.133391+0.819650	test-mae:392.238304+7.776847 
    [575]	train-mae:378.588651+0.816883	test-mae:391.726322+7.774132 
    [576]	train-mae:378.054898+0.818011	test-mae:391.225523+7.774304 
    [577]	train-mae:377.525350+0.821602	test-mae:390.722049+7.759329 
    [578]	train-mae:377.003236+0.822254	test-mae:390.238618+7.757383 
    [579]	train-mae:376.477782+0.823159	test-mae:389.746263+7.756048 
    [580]	train-mae:375.966035+0.828521	test-mae:389.266101+7.750081 
    [581]	train-mae:375.463764+0.829525	test-mae:388.797336+7.740898 
    [582]	train-mae:374.966952+0.829142	test-mae:388.330056+7.739928 
    [583]	train-mae:374.470218+0.833418	test-mae:387.858698+7.730093 
    [584]	train-mae:373.981568+0.831483	test-mae:387.390533+7.727921 
    [585]	train-mae:373.496521+0.828853	test-mae:386.930599+7.725588 
    [586]	train-mae:373.018709+0.830573	test-mae:386.483051+7.720866 
    [587]	train-mae:372.548911+0.830498	test-mae:386.035622+7.717146 
    [588]	train-mae:372.081377+0.834185	test-mae:385.599459+7.701195 
    [589]	train-mae:371.611959+0.837761	test-mae:385.156409+7.693504 
    [590]	train-mae:371.154644+0.836295	test-mae:384.730062+7.687228 
    [591]	train-mae:370.698202+0.836928	test-mae:384.305112+7.684851 
    [592]	train-mae:370.248512+0.838504	test-mae:383.881843+7.678299 
    [593]	train-mae:369.807515+0.840472	test-mae:383.468542+7.670704 
    [594]	train-mae:369.368357+0.845365	test-mae:383.057115+7.657659 
    [595]	train-mae:368.935045+0.846686	test-mae:382.655134+7.648434 
    [596]	train-mae:368.507417+0.846913	test-mae:382.256340+7.639893 
    [597]	train-mae:368.079254+0.844590	test-mae:381.860544+7.630088 
    [598]	train-mae:367.658780+0.846151	test-mae:381.472358+7.629839 
    [599]	train-mae:367.249806+0.845877	test-mae:381.085795+7.622909 
    [600]	train-mae:366.838542+0.844116	test-mae:380.699753+7.610020 
    [601]	train-mae:366.434255+0.841064	test-mae:380.324083+7.604395 
    [602]	train-mae:366.030420+0.839392	test-mae:379.950757+7.604355 
    [603]	train-mae:365.633537+0.840962	test-mae:379.576962+7.592085 
    [604]	train-mae:365.246106+0.838822	test-mae:379.225083+7.585077 
    [605]	train-mae:364.853723+0.844267	test-mae:378.859476+7.575816 
    [606]	train-mae:364.473211+0.846549	test-mae:378.507006+7.577333 
    [607]	train-mae:364.095079+0.848233	test-mae:378.154307+7.567208 
    [608]	train-mae:363.715460+0.845948	test-mae:377.801665+7.561128 
    [609]	train-mae:363.342946+0.842488	test-mae:377.459908+7.554839 
    [610]	train-mae:362.970466+0.842659	test-mae:377.114002+7.552737 
    [611]	train-mae:362.610551+0.841353	test-mae:376.787060+7.544157 
    [612]	train-mae:362.250716+0.841214	test-mae:376.455465+7.536538 
    [613]	train-mae:361.893707+0.836409	test-mae:376.129736+7.537625 
    [614]	train-mae:361.542401+0.841151	test-mae:375.801048+7.536492 
    [615]	train-mae:361.188599+0.842773	test-mae:375.472422+7.525693 
    [616]	train-mae:360.846543+0.844381	test-mae:375.155045+7.519345 
    [617]	train-mae:360.503143+0.844527	test-mae:374.842685+7.511380 
    [618]	train-mae:360.171712+0.851620	test-mae:374.544906+7.502337 
    [619]	train-mae:359.841931+0.851612	test-mae:374.236518+7.503211 
    [620]	train-mae:359.515452+0.849813	test-mae:373.943636+7.494135 
    [621]	train-mae:359.188312+0.848312	test-mae:373.647555+7.497648 
    [622]	train-mae:358.863452+0.847654	test-mae:373.346560+7.488669 
    [623]	train-mae:358.547391+0.850003	test-mae:373.054600+7.480648 
    [624]	train-mae:358.232709+0.851135	test-mae:372.762598+7.469757 
    [625]	train-mae:357.920183+0.848071	test-mae:372.475113+7.464010 
    [626]	train-mae:357.612764+0.855439	test-mae:372.197213+7.457024 
    [627]	train-mae:357.305684+0.859170	test-mae:371.915995+7.455177 
    [628]	train-mae:357.004278+0.861703	test-mae:371.637774+7.453661 
    [629]	train-mae:356.710298+0.866595	test-mae:371.373698+7.452443 
    [630]	train-mae:356.421235+0.870305	test-mae:371.118652+7.447678 
    [631]	train-mae:356.128560+0.873729	test-mae:370.857433+7.442782 
    [632]	train-mae:355.842590+0.875847	test-mae:370.596076+7.438770 
    [633]	train-mae:355.559766+0.876007	test-mae:370.334766+7.435279 
    [634]	train-mae:355.278942+0.875830	test-mae:370.082610+7.431352 
    [635]	train-mae:355.003454+0.873531	test-mae:369.828739+7.424393 
    [636]	train-mae:354.729592+0.874295	test-mae:369.586963+7.425218 
    [637]	train-mae:354.458047+0.873982	test-mae:369.357673+7.422549 
    [638]	train-mae:354.192323+0.871423	test-mae:369.114469+7.426515 
    [639]	train-mae:353.922137+0.875444	test-mae:368.870709+7.416482 
    [640]	train-mae:353.657311+0.880268	test-mae:368.635558+7.407477 
    [641]	train-mae:353.395442+0.879083	test-mae:368.397003+7.410184 
    [642]	train-mae:353.140147+0.879120	test-mae:368.168385+7.409855 
    [643]	train-mae:352.882602+0.879948	test-mae:367.938094+7.406478 
    [644]	train-mae:352.633337+0.882625	test-mae:367.714305+7.398155 
    [645]	train-mae:352.384173+0.883443	test-mae:367.488825+7.395755 
    [646]	train-mae:352.139063+0.882714	test-mae:367.261424+7.399636 
    [647]	train-mae:351.895044+0.880680	test-mae:367.046243+7.397207 
    [648]	train-mae:351.653144+0.877931	test-mae:366.836622+7.390126 
    [649]	train-mae:351.418521+0.880727	test-mae:366.629087+7.384624 
    [650]	train-mae:351.190504+0.883081	test-mae:366.429571+7.387708 
    [651]	train-mae:350.958871+0.882573	test-mae:366.217697+7.390016 
    [652]	train-mae:350.731562+0.887345	test-mae:366.016076+7.395961 
    [653]	train-mae:350.506902+0.886390	test-mae:365.819583+7.398118 
    [654]	train-mae:350.283427+0.886812	test-mae:365.630123+7.395979 
    [655]	train-mae:350.068515+0.887591	test-mae:365.444058+7.406953 
    [656]	train-mae:349.850112+0.887917	test-mae:365.255045+7.406974 
    [657]	train-mae:349.634231+0.892217	test-mae:365.067116+7.409268 
    [658]	train-mae:349.417227+0.890353	test-mae:364.878449+7.411007 
    [659]	train-mae:349.207246+0.890145	test-mae:364.694367+7.416349 
    [660]	train-mae:349.002057+0.890797	test-mae:364.517663+7.414851 
    [661]	train-mae:348.796657+0.893257	test-mae:364.343322+7.416366 
    [662]	train-mae:348.592286+0.895366	test-mae:364.163627+7.425658 
    [663]	train-mae:348.390439+0.898397	test-mae:363.992639+7.427529 
    [664]	train-mae:348.191023+0.903546	test-mae:363.820559+7.425044 
    [665]	train-mae:347.993081+0.905343	test-mae:363.651853+7.427120 
    [666]	train-mae:347.795443+0.908133	test-mae:363.481255+7.424867 
    [667]	train-mae:347.599520+0.907868	test-mae:363.309677+7.419911 
    [668]	train-mae:347.407605+0.905722	test-mae:363.144471+7.423637 
    [669]	train-mae:347.218510+0.905027	test-mae:362.984262+7.436216 
    [670]	train-mae:347.037517+0.902225	test-mae:362.822760+7.442683 
    [671]	train-mae:346.855219+0.902325	test-mae:362.663071+7.451303 
    [672]	train-mae:346.668469+0.909542	test-mae:362.508544+7.452502 
    [673]	train-mae:346.487322+0.910122	test-mae:362.354193+7.453609 
    [674]	train-mae:346.309514+0.912075	test-mae:362.209354+7.450221 
    [675]	train-mae:346.133301+0.911635	test-mae:362.057217+7.454799 
    [676]	train-mae:345.955802+0.911906	test-mae:361.907668+7.458111 
    [677]	train-mae:345.789979+0.911486	test-mae:361.760906+7.461024 
    [678]	train-mae:345.618953+0.912151	test-mae:361.608195+7.460225 
    [679]	train-mae:345.444266+0.914715	test-mae:361.456064+7.461752 
    [680]	train-mae:345.277023+0.918765	test-mae:361.315620+7.464277 
    [681]	train-mae:345.111501+0.923918	test-mae:361.167548+7.465610 
    [682]	train-mae:344.952868+0.926781	test-mae:361.033457+7.469259 
    [683]	train-mae:344.793254+0.927537	test-mae:360.890363+7.474839 
    [684]	train-mae:344.641514+0.926567	test-mae:360.760570+7.480454 
    [685]	train-mae:344.487279+0.925852	test-mae:360.625286+7.487150 
    [686]	train-mae:344.327477+0.925818	test-mae:360.489046+7.485580 
    [687]	train-mae:344.176020+0.930829	test-mae:360.362799+7.485593 
    [688]	train-mae:344.021562+0.931758	test-mae:360.232743+7.490203 
    [689]	train-mae:343.870113+0.932712	test-mae:360.109951+7.506108 
    [690]	train-mae:343.719401+0.931786	test-mae:359.981039+7.510162 
    [691]	train-mae:343.571312+0.927319	test-mae:359.854555+7.512095 
    [692]	train-mae:343.428052+0.927275	test-mae:359.725221+7.512636 
    [693]	train-mae:343.286986+0.925475	test-mae:359.604257+7.509383 
    [694]	train-mae:343.145474+0.932786	test-mae:359.483999+7.509297 
    [695]	train-mae:343.001228+0.934120	test-mae:359.371227+7.508982 
    [696]	train-mae:342.866165+0.931024	test-mae:359.263291+7.506906 
    [697]	train-mae:342.723003+0.938732	test-mae:359.146353+7.495858 
    [698]	train-mae:342.587154+0.936980	test-mae:359.039932+7.499058 
    [699]	train-mae:342.455367+0.938566	test-mae:358.931986+7.498484 
    [700]	train-mae:342.322385+0.940933	test-mae:358.821779+7.494630 
    [701]	train-mae:342.189854+0.939151	test-mae:358.715614+7.494292 
    [702]	train-mae:342.058490+0.942684	test-mae:358.607279+7.496242 
    [703]	train-mae:341.926349+0.945502	test-mae:358.495953+7.497796 
    [704]	train-mae:341.795330+0.945128	test-mae:358.386717+7.506998 
    [705]	train-mae:341.670516+0.951324	test-mae:358.282610+7.500844 
    [706]	train-mae:341.548223+0.956963	test-mae:358.182289+7.504086 
    [707]	train-mae:341.422139+0.958938	test-mae:358.079569+7.499630 
    [708]	train-mae:341.295729+0.960678	test-mae:357.979107+7.505390 
    [709]	train-mae:341.174757+0.957797	test-mae:357.882332+7.510791 
    [710]	train-mae:341.049429+0.958688	test-mae:357.782474+7.519049 
    [711]	train-mae:340.929777+0.962949	test-mae:357.694754+7.515256 
    [712]	train-mae:340.812879+0.961627	test-mae:357.597628+7.517251 
    [713]	train-mae:340.695940+0.959311	test-mae:357.502709+7.524754 
    [714]	train-mae:340.579775+0.962325	test-mae:357.410992+7.526687 
    [715]	train-mae:340.465211+0.961038	test-mae:357.318175+7.537653 
    [716]	train-mae:340.348571+0.963686	test-mae:357.229124+7.532745 
    [717]	train-mae:340.237936+0.962534	test-mae:357.144376+7.536930 
    [718]	train-mae:340.124002+0.966735	test-mae:357.055937+7.543900 
    [719]	train-mae:340.013331+0.968862	test-mae:356.967621+7.548280 
    [720]	train-mae:339.904246+0.968100	test-mae:356.882907+7.550739 
    [721]	train-mae:339.798097+0.970660	test-mae:356.796650+7.554346 
    [722]	train-mae:339.686234+0.973442	test-mae:356.711512+7.554299 
    [723]	train-mae:339.579758+0.976292	test-mae:356.630351+7.554127 
    [724]	train-mae:339.476032+0.980672	test-mae:356.552865+7.561998 
    [725]	train-mae:339.376257+0.978573	test-mae:356.469598+7.564853 
    [726]	train-mae:339.275113+0.977390	test-mae:356.389528+7.562570 
    [727]	train-mae:339.173722+0.980758	test-mae:356.310898+7.555871 
    [728]	train-mae:339.075466+0.982918	test-mae:356.238718+7.557169 
    [729]	train-mae:338.976989+0.987391	test-mae:356.164404+7.554328 
    [730]	train-mae:338.876776+0.983652	test-mae:356.091395+7.557130 
    [731]	train-mae:338.782000+0.984356	test-mae:356.014238+7.561244 
    [732]	train-mae:338.684803+0.985894	test-mae:355.938998+7.563003 
    [733]	train-mae:338.586063+0.987272	test-mae:355.863048+7.564272 
    [734]	train-mae:338.492151+0.993133	test-mae:355.791946+7.565488 
    [735]	train-mae:338.397261+0.995990	test-mae:355.717515+7.559722 
    [736]	train-mae:338.305871+1.000828	test-mae:355.644094+7.559019 
    [737]	train-mae:338.213916+0.995895	test-mae:355.571400+7.563741 
    [738]	train-mae:338.125256+0.996734	test-mae:355.503710+7.565796 
    [739]	train-mae:338.036348+0.999257	test-mae:355.436113+7.567151 
    [740]	train-mae:337.950279+0.997829	test-mae:355.372210+7.573671 
    [741]	train-mae:337.859866+1.003413	test-mae:355.303370+7.580298 
    [742]	train-mae:337.772707+1.004327	test-mae:355.247613+7.589131 
    [743]	train-mae:337.689198+1.008339	test-mae:355.176305+7.586301 
    [744]	train-mae:337.603470+1.015874	test-mae:355.115021+7.581949 
    [745]	train-mae:337.517947+1.021064	test-mae:355.049796+7.586020 
    [746]	train-mae:337.437157+1.019084	test-mae:354.991500+7.592548 
    [747]	train-mae:337.354328+1.019552	test-mae:354.934146+7.595621 
    [748]	train-mae:337.268792+1.018429	test-mae:354.869227+7.595863 
    [749]	train-mae:337.187290+1.019853	test-mae:354.810193+7.596690 
    [750]	train-mae:337.104934+1.019435	test-mae:354.751468+7.594478 
    [751]	train-mae:337.023089+1.017239	test-mae:354.690919+7.593306 
    [752]	train-mae:336.943085+1.017117	test-mae:354.632659+7.593481 
    [753]	train-mae:336.866507+1.016499	test-mae:354.578227+7.603092 
    [754]	train-mae:336.784169+1.017998	test-mae:354.520668+7.602682 
    [755]	train-mae:336.705319+1.019246	test-mae:354.465860+7.607935 
    [756]	train-mae:336.628001+1.021880	test-mae:354.409067+7.607473 
    [757]	train-mae:336.549194+1.023363	test-mae:354.359547+7.605018 
    [758]	train-mae:336.470993+1.025451	test-mae:354.301645+7.602223 
    [759]	train-mae:336.393757+1.027482	test-mae:354.249661+7.603782 
    [760]	train-mae:336.317542+1.025730	test-mae:354.190780+7.603392 
    [761]	train-mae:336.242291+1.028776	test-mae:354.139122+7.604296 
    [762]	train-mae:336.169060+1.027575	test-mae:354.088079+7.601828 
    [763]	train-mae:336.097657+1.028659	test-mae:354.034769+7.607270 
    [764]	train-mae:336.023236+1.029117	test-mae:353.984275+7.613138 
    [765]	train-mae:335.950631+1.029105	test-mae:353.936426+7.612833 
    [766]	train-mae:335.878906+1.031406	test-mae:353.884377+7.614008 
    [767]	train-mae:335.811031+1.031577	test-mae:353.833648+7.615862 
    [768]	train-mae:335.743806+1.034136	test-mae:353.779410+7.613339 
    [769]	train-mae:335.675171+1.038083	test-mae:353.733067+7.616288 
    [770]	train-mae:335.605580+1.040447	test-mae:353.686559+7.621243 
    [771]	train-mae:335.538543+1.035841	test-mae:353.642978+7.627988 
    [772]	train-mae:335.472315+1.040279	test-mae:353.597236+7.627313 
    [773]	train-mae:335.406525+1.043603	test-mae:353.561579+7.628396 
    [774]	train-mae:335.340074+1.041271	test-mae:353.522877+7.624231 
    [775]	train-mae:335.278109+1.041962	test-mae:353.480542+7.623195 
    [776]	train-mae:335.216640+1.044509	test-mae:353.441487+7.619523 
    [777]	train-mae:335.153696+1.045488	test-mae:353.400163+7.622169 
    [778]	train-mae:335.090702+1.046154	test-mae:353.353633+7.629491 
    [779]	train-mae:335.028155+1.046073	test-mae:353.323460+7.632895 
    [780]	train-mae:334.965938+1.049267	test-mae:353.280889+7.630452 
    [781]	train-mae:334.907797+1.048264	test-mae:353.240765+7.637376 
    [782]	train-mae:334.848956+1.049134	test-mae:353.203076+7.646157 
    [783]	train-mae:334.786659+1.046359	test-mae:353.160058+7.646787 
    [784]	train-mae:334.727039+1.049502	test-mae:353.121110+7.642582 
    [785]	train-mae:334.667206+1.050588	test-mae:353.081110+7.646352 
    [786]	train-mae:334.610497+1.054697	test-mae:353.046137+7.646757 
    [787]	train-mae:334.549357+1.051734	test-mae:353.006843+7.654271 
    [788]	train-mae:334.497022+1.053298	test-mae:352.977297+7.656147 
    [789]	train-mae:334.440362+1.057532	test-mae:352.937305+7.652888 
    [790]	train-mae:334.381553+1.053757	test-mae:352.896314+7.665880 
    [791]	train-mae:334.325639+1.054209	test-mae:352.858704+7.656097 
    [792]	train-mae:334.267481+1.053270	test-mae:352.822671+7.660752 
    [793]	train-mae:334.211919+1.054985	test-mae:352.785371+7.661634 
    [794]	train-mae:334.159436+1.056581	test-mae:352.757648+7.667122 
    [795]	train-mae:334.103924+1.059082	test-mae:352.722845+7.668135 
    [796]	train-mae:334.051220+1.058342	test-mae:352.693056+7.671919 
    [797]	train-mae:333.997843+1.055189	test-mae:352.662000+7.675637 
    [798]	train-mae:333.940966+1.054393	test-mae:352.630866+7.672844 
    [799]	train-mae:333.886664+1.052631	test-mae:352.595050+7.671964 
    [800]	train-mae:333.834867+1.051174	test-mae:352.567675+7.669512 
    [801]	train-mae:333.781576+1.052270	test-mae:352.534099+7.664061 
    [802]	train-mae:333.728377+1.055269	test-mae:352.499832+7.660292 
    [803]	train-mae:333.677960+1.055110	test-mae:352.475747+7.661901 
    [804]	train-mae:333.632040+1.051672	test-mae:352.449137+7.669966 
    [805]	train-mae:333.583197+1.053249	test-mae:352.419843+7.670616 
    [806]	train-mae:333.534184+1.056713	test-mae:352.393175+7.667678 
    [807]	train-mae:333.484370+1.058078	test-mae:352.362578+7.665944 
    [808]	train-mae:333.434106+1.058660	test-mae:352.334602+7.664216 
    [809]	train-mae:333.387617+1.060500	test-mae:352.306077+7.664854 
    [810]	train-mae:333.334991+1.058100	test-mae:352.276311+7.663993 
    [811]	train-mae:333.285280+1.061076	test-mae:352.247241+7.658207 
    [812]	train-mae:333.235844+1.062299	test-mae:352.217213+7.657057 
    [813]	train-mae:333.188891+1.064282	test-mae:352.194378+7.654467 
    [814]	train-mae:333.143150+1.064022	test-mae:352.163660+7.654140 
    [815]	train-mae:333.102205+1.065986	test-mae:352.143367+7.647132 
    [816]	train-mae:333.054196+1.067742	test-mae:352.117683+7.647606 
    [817]	train-mae:333.008443+1.069531	test-mae:352.092947+7.646692 
    [818]	train-mae:332.966560+1.075385	test-mae:352.074230+7.641712 
    [819]	train-mae:332.920804+1.076234	test-mae:352.047834+7.646291 
    [820]	train-mae:332.877977+1.078430	test-mae:352.024224+7.646023 
    [821]	train-mae:332.830644+1.077911	test-mae:351.997426+7.644018 
    [822]	train-mae:332.788404+1.080308	test-mae:351.968003+7.642804 
    [823]	train-mae:332.743797+1.081954	test-mae:351.941101+7.642144 
    [824]	train-mae:332.701953+1.081580	test-mae:351.915188+7.643140 
    [825]	train-mae:332.655567+1.081209	test-mae:351.884611+7.645241 
    [826]	train-mae:332.618394+1.081887	test-mae:351.861016+7.650661 
    [827]	train-mae:332.574609+1.084734	test-mae:351.833941+7.655226 
    [828]	train-mae:332.532374+1.085440	test-mae:351.806602+7.654997 
    [829]	train-mae:332.493320+1.086922	test-mae:351.789409+7.655770 
    [830]	train-mae:332.452092+1.087311	test-mae:351.770517+7.650504 
    [831]	train-mae:332.409786+1.084993	test-mae:351.741230+7.656673 
    [832]	train-mae:332.366469+1.087681	test-mae:351.715693+7.654681 
    [833]	train-mae:332.324513+1.084685	test-mae:351.698039+7.659444 
    [834]	train-mae:332.286593+1.083192	test-mae:351.676797+7.665402 
    [835]	train-mae:332.244727+1.087652	test-mae:351.657846+7.661753 
    [836]	train-mae:332.209721+1.087293	test-mae:351.640611+7.663846 
    [837]	train-mae:332.170676+1.089362	test-mae:351.618744+7.665385 
    [838]	train-mae:332.131688+1.090593	test-mae:351.605184+7.661062 
    [839]	train-mae:332.093739+1.091088	test-mae:351.584534+7.653745 
    [840]	train-mae:332.057402+1.091214	test-mae:351.565602+7.656362 
    [841]	train-mae:332.025396+1.089181	test-mae:351.550299+7.655761 
    [842]	train-mae:331.990811+1.087909	test-mae:351.532714+7.659440 
    [843]	train-mae:331.949613+1.088086	test-mae:351.511190+7.654287 
    [844]	train-mae:331.912891+1.090446	test-mae:351.494539+7.647223 
    [845]	train-mae:331.876425+1.089940	test-mae:351.479371+7.647871 
    [846]	train-mae:331.843003+1.094940	test-mae:351.461349+7.645333 
    [847]	train-mae:331.805382+1.095413	test-mae:351.441175+7.650813 
    [848]	train-mae:331.770672+1.092153	test-mae:351.425841+7.648419 
    [849]	train-mae:331.731921+1.092020	test-mae:351.406517+7.652654 
    [850]	train-mae:331.698158+1.091871	test-mae:351.388850+7.654810 
    [851]	train-mae:331.666134+1.094005	test-mae:351.370125+7.654395 
    [852]	train-mae:331.628230+1.091207	test-mae:351.353843+7.663735 
    [853]	train-mae:331.593932+1.094386	test-mae:351.341305+7.664735 
    [854]	train-mae:331.559192+1.097102	test-mae:351.336035+7.662330 
    [855]	train-mae:331.525209+1.097700	test-mae:351.315360+7.663219 
    [856]	train-mae:331.491629+1.097208	test-mae:351.298398+7.666146 
    [857]	train-mae:331.462036+1.100458	test-mae:351.284165+7.666636 
    [858]	train-mae:331.427059+1.106410	test-mae:351.267319+7.666117 
    [859]	train-mae:331.395950+1.104008	test-mae:351.257007+7.667228 
    [860]	train-mae:331.365303+1.102716	test-mae:351.246556+7.668890 
    [861]	train-mae:331.335055+1.104896	test-mae:351.232684+7.674514 
    [862]	train-mae:331.304550+1.106167	test-mae:351.218554+7.668260 
    [863]	train-mae:331.273054+1.106447	test-mae:351.205508+7.660315 
    [864]	train-mae:331.242359+1.111611	test-mae:351.191459+7.660668 
    [865]	train-mae:331.211966+1.110331	test-mae:351.175831+7.666047 
    [866]	train-mae:331.179523+1.115535	test-mae:351.160778+7.669153 
    [867]	train-mae:331.149198+1.113519	test-mae:351.148692+7.674244 
    [868]	train-mae:331.117219+1.116721	test-mae:351.134408+7.670614 
    [869]	train-mae:331.090927+1.123298	test-mae:351.128016+7.671457 
    [870]	train-mae:331.058063+1.128655	test-mae:351.117569+7.669283 
    [871]	train-mae:331.029218+1.128471	test-mae:351.103757+7.672211 
    [872]	train-mae:330.998981+1.129037	test-mae:351.085104+7.676251 
    [873]	train-mae:330.968451+1.125905	test-mae:351.072905+7.675894 
    [874]	train-mae:330.938103+1.132268	test-mae:351.060274+7.672885 
    [875]	train-mae:330.905934+1.128441	test-mae:351.053382+7.676934 
    [876]	train-mae:330.873278+1.126408	test-mae:351.040611+7.680954 
    [877]	train-mae:330.841939+1.123691	test-mae:351.036803+7.685822 
    [878]	train-mae:330.811988+1.120283	test-mae:351.023752+7.682330 
    [879]	train-mae:330.778283+1.124029	test-mae:351.009745+7.681041 
    [880]	train-mae:330.747117+1.122436	test-mae:350.999060+7.682707 
    [881]	train-mae:330.717397+1.118520	test-mae:350.987004+7.685999 
    [882]	train-mae:330.688641+1.124578	test-mae:350.970498+7.681892 
    [883]	train-mae:330.660727+1.125359	test-mae:350.955878+7.686645 
    [884]	train-mae:330.635818+1.121833	test-mae:350.944511+7.686588 
    [885]	train-mae:330.604789+1.119060	test-mae:350.934173+7.689240 
    [886]	train-mae:330.575204+1.120139	test-mae:350.930319+7.688883 
    [887]	train-mae:330.551124+1.118077	test-mae:350.920169+7.688697 
    [888]	train-mae:330.521464+1.117810	test-mae:350.912203+7.687862 
    [889]	train-mae:330.493801+1.120336	test-mae:350.903556+7.691423 
    [890]	train-mae:330.467306+1.121122	test-mae:350.891036+7.690618 
    [891]	train-mae:330.439692+1.117675	test-mae:350.887750+7.692419 
    [892]	train-mae:330.410923+1.118370	test-mae:350.879629+7.695465 
    [893]	train-mae:330.381590+1.113984	test-mae:350.881812+7.698788 
    [894]	train-mae:330.358680+1.114266	test-mae:350.869735+7.698870 
    [895]	train-mae:330.332351+1.117216	test-mae:350.863370+7.701649 
    [896]	train-mae:330.307188+1.118611	test-mae:350.852709+7.703028 
    [897]	train-mae:330.285263+1.116999	test-mae:350.845302+7.701891 
    [898]	train-mae:330.260759+1.117237	test-mae:350.834948+7.700851 
    [899]	train-mae:330.236219+1.116806	test-mae:350.825651+7.703127 
    [900]	train-mae:330.207925+1.118073	test-mae:350.824793+7.698581 
    [901]	train-mae:330.184627+1.120709	test-mae:350.812684+7.701583 
    [902]	train-mae:330.161074+1.122351	test-mae:350.802767+7.698991 
    [903]	train-mae:330.132867+1.122700	test-mae:350.800892+7.709173 
    [904]	train-mae:330.110083+1.123583	test-mae:350.798607+7.710964 
    [905]	train-mae:330.087611+1.122493	test-mae:350.789404+7.718080 
    [906]	train-mae:330.062257+1.124082	test-mae:350.780911+7.719171 
    [907]	train-mae:330.037668+1.123603	test-mae:350.773772+7.720337 
    [908]	train-mae:330.014872+1.123160	test-mae:350.774217+7.719767 
    [909]	train-mae:329.991487+1.124811	test-mae:350.765463+7.721914 
    [910]	train-mae:329.966836+1.122115	test-mae:350.757254+7.723752 
    [911]	train-mae:329.944302+1.118483	test-mae:350.746096+7.723767 
    [912]	train-mae:329.916355+1.119685	test-mae:350.737387+7.724449 
    [913]	train-mae:329.893970+1.120154	test-mae:350.729141+7.729984 
    [914]	train-mae:329.867655+1.118881	test-mae:350.722866+7.725868 
    [915]	train-mae:329.844974+1.123245	test-mae:350.709854+7.719414 
    [916]	train-mae:329.820976+1.117876	test-mae:350.705713+7.720757 
    [917]	train-mae:329.794668+1.121948	test-mae:350.698157+7.718338 
    [918]	train-mae:329.772028+1.123680	test-mae:350.690965+7.719310 
    [919]	train-mae:329.749766+1.119179	test-mae:350.685343+7.723715 
    [920]	train-mae:329.727969+1.120239	test-mae:350.672884+7.722687 
    [921]	train-mae:329.702333+1.121118	test-mae:350.669222+7.723245 
    [922]	train-mae:329.685237+1.119948	test-mae:350.660212+7.726692 
    [923]	train-mae:329.665516+1.117511	test-mae:350.655993+7.725364 
    [924]	train-mae:329.644805+1.115473	test-mae:350.647044+7.724500 
    [925]	train-mae:329.618182+1.114176	test-mae:350.642419+7.729647 
    [926]	train-mae:329.595400+1.118703	test-mae:350.635042+7.731295 
    [927]	train-mae:329.573430+1.120651	test-mae:350.626725+7.730898 
    [928]	train-mae:329.550750+1.121303	test-mae:350.622725+7.730833 
    [929]	train-mae:329.528694+1.116975	test-mae:350.615044+7.734995 
    [930]	train-mae:329.505487+1.117261	test-mae:350.614441+7.732687 
    [931]	train-mae:329.484375+1.118652	test-mae:350.614072+7.730457 
    [932]	train-mae:329.458580+1.123018	test-mae:350.607357+7.726217 
    [933]	train-mae:329.437089+1.120912	test-mae:350.604656+7.726935 
    [934]	train-mae:329.417461+1.118594	test-mae:350.602901+7.731547 
    [935]	train-mae:329.396864+1.115942	test-mae:350.598743+7.734878 
    [936]	train-mae:329.378131+1.115129	test-mae:350.594362+7.729282 
    [937]	train-mae:329.359645+1.114593	test-mae:350.589867+7.734330 
    [938]	train-mae:329.339808+1.117948	test-mae:350.589127+7.739009 
    [939]	train-mae:329.319855+1.110587	test-mae:350.586188+7.743567 
    [940]	train-mae:329.296337+1.111514	test-mae:350.581717+7.739442 
    [941]	train-mae:329.273464+1.109041	test-mae:350.578540+7.744298 
    [942]	train-mae:329.249308+1.106541	test-mae:350.569865+7.745376 
    [943]	train-mae:329.230387+1.104809	test-mae:350.568573+7.747619 
    [944]	train-mae:329.210919+1.108124	test-mae:350.565271+7.744454 
    [945]	train-mae:329.189266+1.107304	test-mae:350.563241+7.739738 
    [946]	train-mae:329.167709+1.106753	test-mae:350.560623+7.744011 
    [947]	train-mae:329.147676+1.101996	test-mae:350.563251+7.745975 
    [948]	train-mae:329.127806+1.105342	test-mae:350.558667+7.748212 
    [949]	train-mae:329.108284+1.105340	test-mae:350.549347+7.747050 
    [950]	train-mae:329.090360+1.106962	test-mae:350.543270+7.745038 
    [951]	train-mae:329.071704+1.106254	test-mae:350.537662+7.743428 
    [952]	train-mae:329.055355+1.103675	test-mae:350.532685+7.742786 
    [953]	train-mae:329.039143+1.103306	test-mae:350.529368+7.743239 
    [954]	train-mae:329.020507+1.108145	test-mae:350.524775+7.744265 
    [955]	train-mae:328.997191+1.108173	test-mae:350.518976+7.747473 
    [956]	train-mae:328.973911+1.108652	test-mae:350.522100+7.748154 
    [957]	train-mae:328.953461+1.109413	test-mae:350.515936+7.748730 
    [958]	train-mae:328.931703+1.107943	test-mae:350.516756+7.746256 
    [959]	train-mae:328.913060+1.105795	test-mae:350.517743+7.745881 
    [960]	train-mae:328.893975+1.103806	test-mae:350.507286+7.742041 
    [961]	train-mae:328.872242+1.105333	test-mae:350.501463+7.747993 
    [962]	train-mae:328.850980+1.106081	test-mae:350.492762+7.746995 
    [963]	train-mae:328.834958+1.104720	test-mae:350.493052+7.747669 
    [964]	train-mae:328.818398+1.101238	test-mae:350.487072+7.757124 
    [965]	train-mae:328.802584+1.097483	test-mae:350.482459+7.760939 
    [966]	train-mae:328.777895+1.096412	test-mae:350.480194+7.762187 
    [967]	train-mae:328.755893+1.099743	test-mae:350.475731+7.755823 
    [968]	train-mae:328.739187+1.096370	test-mae:350.474369+7.760940 
    [969]	train-mae:328.721827+1.096849	test-mae:350.474900+7.758854 
    [970]	train-mae:328.703098+1.098769	test-mae:350.473238+7.756617 
    [971]	train-mae:328.682949+1.101727	test-mae:350.471527+7.758516 
    [972]	train-mae:328.668174+1.099036	test-mae:350.474629+7.759068 
    [973]	train-mae:328.643678+1.099817	test-mae:350.474280+7.758954 
    [974]	train-mae:328.623929+1.095912	test-mae:350.477292+7.761811 
    [975]	train-mae:328.599486+1.095408	test-mae:350.473705+7.763282 
    [976]	train-mae:328.582792+1.099921	test-mae:350.469756+7.761526 
    [977]	train-mae:328.562842+1.102266	test-mae:350.466701+7.760123 
    [978]	train-mae:328.543593+1.100377	test-mae:350.463383+7.765607 
    [979]	train-mae:328.524960+1.102205	test-mae:350.463049+7.762617 
    [980]	train-mae:328.507330+1.101878	test-mae:350.460968+7.763474 
    [981]	train-mae:328.490658+1.101312	test-mae:350.459755+7.764404 
    [982]	train-mae:328.471956+1.097714	test-mae:350.456817+7.764886 
    [983]	train-mae:328.451839+1.099142	test-mae:350.453970+7.764376 
    [984]	train-mae:328.436397+1.095946	test-mae:350.457162+7.765444 
    [985]	train-mae:328.418417+1.095590	test-mae:350.458246+7.760845 
    [986]	train-mae:328.400846+1.101512	test-mae:350.455489+7.761742 
    [987]	train-mae:328.383578+1.100404	test-mae:350.452793+7.762183 
    [988]	train-mae:328.364051+1.101493	test-mae:350.450551+7.768841 
    [989]	train-mae:328.339581+1.101596	test-mae:350.447975+7.761109 
    [990]	train-mae:328.322025+1.098497	test-mae:350.448346+7.761079 
    [991]	train-mae:328.300430+1.097532	test-mae:350.447099+7.757889 
    [992]	train-mae:328.282503+1.097930	test-mae:350.441983+7.751627 
    [993]	train-mae:328.264187+1.097588	test-mae:350.440501+7.757249 
    [994]	train-mae:328.249126+1.096794	test-mae:350.437555+7.756858 
    [995]	train-mae:328.230360+1.094779	test-mae:350.427725+7.754086 
    [996]	train-mae:328.211727+1.094283	test-mae:350.424887+7.757916 
    [997]	train-mae:328.193518+1.096435	test-mae:350.423621+7.758615 
    [998]	train-mae:328.176466+1.098960	test-mae:350.417133+7.757135 
    [999]	train-mae:328.156570+1.101594	test-mae:350.415336+7.754029 
    [1000]	train-mae:328.138178+1.105720	test-mae:350.419333+7.756010 
    [1001]	train-mae:328.121344+1.104733	test-mae:350.415486+7.760500 
    [1002]	train-mae:328.102190+1.107960	test-mae:350.413976+7.754299 
    [1003]	train-mae:328.083218+1.105267	test-mae:350.410832+7.756381 
    [1004]	train-mae:328.068600+1.106056	test-mae:350.407259+7.755416 
    [1005]	train-mae:328.049737+1.109232	test-mae:350.401223+7.752042 
    [1006]	train-mae:328.030390+1.107760	test-mae:350.401846+7.755243 
    [1007]	train-mae:328.014543+1.109194	test-mae:350.403372+7.756603 
    [1008]	train-mae:327.997750+1.108915	test-mae:350.400021+7.757455 
    [1009]	train-mae:327.981456+1.108386	test-mae:350.397675+7.754603 
    [1010]	train-mae:327.961689+1.109088	test-mae:350.396385+7.756593 
    [1011]	train-mae:327.941768+1.111412	test-mae:350.396185+7.752328 
    [1012]	train-mae:327.923161+1.114822	test-mae:350.389880+7.752064 
    [1013]	train-mae:327.904700+1.115087	test-mae:350.387506+7.753285 
    [1014]	train-mae:327.886241+1.113659	test-mae:350.385691+7.752094 
    [1015]	train-mae:327.871412+1.114672	test-mae:350.386261+7.748651 
    [1016]	train-mae:327.853457+1.119437	test-mae:350.380873+7.746077 
    [1017]	train-mae:327.836498+1.121810	test-mae:350.381288+7.743845 
    [1018]	train-mae:327.823275+1.124653	test-mae:350.382073+7.746840 
    [1019]	train-mae:327.807396+1.126910	test-mae:350.380211+7.743210 
    [1020]	train-mae:327.793061+1.131051	test-mae:350.379177+7.746539 
    [1021]	train-mae:327.775362+1.127613	test-mae:350.379667+7.746731 
    [1022]	train-mae:327.758456+1.129206	test-mae:350.376658+7.748786 
    [1023]	train-mae:327.741050+1.130868	test-mae:350.376068+7.751392 
    [1024]	train-mae:327.723078+1.125628	test-mae:350.379400+7.754851 
    [1025]	train-mae:327.711296+1.125756	test-mae:350.388061+7.752734 
    [1026]	train-mae:327.696631+1.130999	test-mae:350.393551+7.754501 
    [1027]	train-mae:327.680792+1.134555	test-mae:350.395035+7.753444 
    [1028]	train-mae:327.667054+1.134693	test-mae:350.392203+7.755033 
    [1029]	train-mae:327.651392+1.131394	test-mae:350.392932+7.759007 
    [1030]	train-mae:327.638289+1.126800	test-mae:350.395406+7.753761 
    [1031]	train-mae:327.620166+1.127513	test-mae:350.395375+7.746236 
    [1032]	train-mae:327.607669+1.128014	test-mae:350.396858+7.741838 
    [1033]	train-mae:327.591997+1.127967	test-mae:350.397616+7.741913 
    [1034]	train-mae:327.577314+1.130800	test-mae:350.392774+7.741151 
    [1035]	train-mae:327.560858+1.130590	test-mae:350.392805+7.742018 
    [1036]	train-mae:327.546792+1.130165	test-mae:350.390668+7.744543 
    [1037]	train-mae:327.531152+1.131357	test-mae:350.390574+7.747932 
    [1038]	train-mae:327.516770+1.130036	test-mae:350.386728+7.747680 
    [1039]	train-mae:327.501578+1.128630	test-mae:350.387817+7.749190 
    Stopping. Best iteration:
    [1023]	train-mae:327.741050+1.130868	test-mae:350.376068+7.751392
    


Use 1,023 rounds to make the model.


```R
set.seed(12)
xgb9 = xgboost(params = list(eta = .006, min_child_weight = 3,
                             subsample = .7, reg_lambda = .5),
               data = train.xgb, nrounds = 1023, max_depth = 5,
               eval_metric = 'mae')
```

    [1]	train-mae:5989.143064 
    [2]	train-mae:5953.241212 
    [3]	train-mae:5917.546423 
    [4]	train-mae:5882.062729 
    [5]	train-mae:5846.824227 
    [6]	train-mae:5811.782146 
    [7]	train-mae:5776.952591 
    [8]	train-mae:5742.318340 
    [9]	train-mae:5707.870097 
    [10]	train-mae:5673.635292 
    [11]	train-mae:5639.639527 
    [12]	train-mae:5605.834113 
    [13]	train-mae:5572.211857 
    [14]	train-mae:5538.842174 
    [15]	train-mae:5505.612335 
    [16]	train-mae:5472.612495 
    [17]	train-mae:5439.792568 
    [18]	train-mae:5407.211229 
    [19]	train-mae:5374.803813 
    [20]	train-mae:5342.563570 
    [21]	train-mae:5310.517885 
    [22]	train-mae:5278.659854 
    [23]	train-mae:5247.034117 
    [24]	train-mae:5215.598266 
    [25]	train-mae:5184.338574 
    [26]	train-mae:5153.253176 
    [27]	train-mae:5122.352123 
    [28]	train-mae:5091.632282 
    [29]	train-mae:5061.159558 
    [30]	train-mae:5030.870029 
    [31]	train-mae:5000.709729 
    [32]	train-mae:4970.743685 
    [33]	train-mae:4940.933013 
    [34]	train-mae:4911.287627 
    [35]	train-mae:4881.841012 
    [36]	train-mae:4852.529199 
    [37]	train-mae:4823.482529 
    [38]	train-mae:4794.565368 
    [39]	train-mae:4765.825709 
    [40]	train-mae:4737.271830 
    [41]	train-mae:4708.912454 
    [42]	train-mae:4680.675402 
    [43]	train-mae:4652.624096 
    [44]	train-mae:4624.728269 
    [45]	train-mae:4596.988720 
    [46]	train-mae:4569.430664 
    [47]	train-mae:4542.046069 
    [48]	train-mae:4514.823462 
    [49]	train-mae:4487.786726 
    [50]	train-mae:4460.892610 
    [51]	train-mae:4434.164959 
    [52]	train-mae:4407.611284 
    [53]	train-mae:4381.164494 
    [54]	train-mae:4354.907564 
    [55]	train-mae:4328.805477 
    [56]	train-mae:4302.849806 
    [57]	train-mae:4277.086704 
    [58]	train-mae:4251.452480 
    [59]	train-mae:4225.942601 
    [60]	train-mae:4200.617454 
    [61]	train-mae:4175.451662 
    [62]	train-mae:4150.403794 
    [63]	train-mae:4125.498789 
    [64]	train-mae:4100.782803 
    [65]	train-mae:4076.235458 
    [66]	train-mae:4051.787078 
    [67]	train-mae:4027.488124 
    [68]	train-mae:4003.352916 
    [69]	train-mae:3979.390836 
    [70]	train-mae:3955.564464 
    [71]	train-mae:3931.846708 
    [72]	train-mae:3908.303746 
    [73]	train-mae:3884.897230 
    [74]	train-mae:3861.636771 
    [75]	train-mae:3838.492324 
    [76]	train-mae:3815.480586 
    [77]	train-mae:3792.635769 
    [78]	train-mae:3769.927062 
    [79]	train-mae:3747.377317 
    [80]	train-mae:3724.922234 
    [81]	train-mae:3702.588093 
    [82]	train-mae:3680.437464 
    [83]	train-mae:3658.400201 
    [84]	train-mae:3636.503758 
    [85]	train-mae:3614.709809 
    [86]	train-mae:3593.087151 
    [87]	train-mae:3571.607321 
    [88]	train-mae:3550.197321 
    [89]	train-mae:3528.942075 
    [90]	train-mae:3507.812704 
    [91]	train-mae:3486.817429 
    [92]	train-mae:3465.957481 
    [93]	train-mae:3445.203820 
    [94]	train-mae:3424.546252 
    [95]	train-mae:3404.065330 
    [96]	train-mae:3383.699251 
    [97]	train-mae:3363.477227 
    [98]	train-mae:3343.329094 
    [99]	train-mae:3323.317345 
    [100]	train-mae:3303.422561 
    [101]	train-mae:3283.697439 
    [102]	train-mae:3264.080312 
    [103]	train-mae:3244.547295 
    [104]	train-mae:3225.178559 
    [105]	train-mae:3205.900427 
    [106]	train-mae:3186.719215 
    [107]	train-mae:3167.672087 
    [108]	train-mae:3148.733060 
    [109]	train-mae:3129.880308 
    [110]	train-mae:3111.150493 
    [111]	train-mae:3092.567047 
    [112]	train-mae:3074.090048 
    [113]	train-mae:3055.698202 
    [114]	train-mae:3037.429265 
    [115]	train-mae:3019.253619 
    [116]	train-mae:3001.224115 
    [117]	train-mae:2983.301349 
    [118]	train-mae:2965.452084 
    [119]	train-mae:2947.732355 
    [120]	train-mae:2930.122927 
    [121]	train-mae:2912.661224 
    [122]	train-mae:2895.307051 
    [123]	train-mae:2878.035456 
    [124]	train-mae:2860.839304 
    [125]	train-mae:2843.769120 
    [126]	train-mae:2826.794512 
    [127]	train-mae:2809.930387 
    [128]	train-mae:2793.182110 
    [129]	train-mae:2776.513040 
    [130]	train-mae:2759.933418 
    [131]	train-mae:2743.476728 
    [132]	train-mae:2727.133953 
    [133]	train-mae:2710.885999 
    [134]	train-mae:2694.725423 
    [135]	train-mae:2678.681691 
    [136]	train-mae:2662.726711 
    [137]	train-mae:2646.850362 
    [138]	train-mae:2631.074916 
    [139]	train-mae:2615.388854 
    [140]	train-mae:2599.778825 
    [141]	train-mae:2584.264668 
    [142]	train-mae:2568.856795 
    [143]	train-mae:2553.526505 
    [144]	train-mae:2538.313309 
    [145]	train-mae:2523.177865 
    [146]	train-mae:2508.187802 
    [147]	train-mae:2493.234634 
    [148]	train-mae:2478.385622 
    [149]	train-mae:2463.637925 
    [150]	train-mae:2448.966211 
    [151]	train-mae:2434.378546 
    [152]	train-mae:2419.903390 
    [153]	train-mae:2405.507537 
    [154]	train-mae:2391.207136 
    [155]	train-mae:2376.959194 
    [156]	train-mae:2362.799766 
    [157]	train-mae:2348.729741 
    [158]	train-mae:2334.780296 
    [159]	train-mae:2320.907071 
    [160]	train-mae:2307.109791 
    [161]	train-mae:2293.381116 
    [162]	train-mae:2279.717198 
    [163]	train-mae:2266.182586 
    [164]	train-mae:2252.689098 
    [165]	train-mae:2239.290438 
    [166]	train-mae:2225.970064 
    [167]	train-mae:2212.751831 
    [168]	train-mae:2199.631476 
    [169]	train-mae:2186.534450 
    [170]	train-mae:2173.551710 
    [171]	train-mae:2160.660122 
    [172]	train-mae:2147.810658 
    [173]	train-mae:2135.068281 
    [174]	train-mae:2122.412872 
    [175]	train-mae:2109.800339 
    [176]	train-mae:2097.274869 
    [177]	train-mae:2084.828165 
    [178]	train-mae:2072.452873 
    [179]	train-mae:2060.155891 
    [180]	train-mae:2047.958428 
    [181]	train-mae:2035.792548 
    [182]	train-mae:2023.696459 
    [183]	train-mae:2011.721984 
    [184]	train-mae:1999.801903 
    [185]	train-mae:1987.955865 
    [186]	train-mae:1976.199754 
    [187]	train-mae:1964.494651 
    [188]	train-mae:1952.824323 
    [189]	train-mae:1941.258270 
    [190]	train-mae:1929.771261 
    [191]	train-mae:1918.334348 
    [192]	train-mae:1907.019991 
    [193]	train-mae:1895.737553 
    [194]	train-mae:1884.539931 
    [195]	train-mae:1873.404753 
    [196]	train-mae:1862.374799 
    [197]	train-mae:1851.373517 
    [198]	train-mae:1840.430786 
    [199]	train-mae:1829.557015 
    [200]	train-mae:1818.778467 
    [201]	train-mae:1808.041617 
    [202]	train-mae:1797.360803 
    [203]	train-mae:1786.776880 
    [204]	train-mae:1776.255347 
    [205]	train-mae:1765.787266 
    [206]	train-mae:1755.383645 
    [207]	train-mae:1745.058320 
    [208]	train-mae:1734.797394 
    [209]	train-mae:1724.574339 
    [210]	train-mae:1714.439903 
    [211]	train-mae:1704.347269 
    [212]	train-mae:1694.305563 
    [213]	train-mae:1684.354275 
    [214]	train-mae:1674.475793 
    [215]	train-mae:1664.633060 
    [216]	train-mae:1654.823385 
    [217]	train-mae:1645.120244 
    [218]	train-mae:1635.457193 
    [219]	train-mae:1625.865955 
    [220]	train-mae:1616.289630 
    [221]	train-mae:1606.803938 
    [222]	train-mae:1597.362215 
    [223]	train-mae:1587.992722 
    [224]	train-mae:1578.684555 
    [225]	train-mae:1569.450452 
    [226]	train-mae:1560.228705 
    [227]	train-mae:1551.098030 
    [228]	train-mae:1542.037510 
    [229]	train-mae:1533.005379 
    [230]	train-mae:1524.052846 
    [231]	train-mae:1515.151333 
    [232]	train-mae:1506.289460 
    [233]	train-mae:1497.501418 
    [234]	train-mae:1488.756732 
    [235]	train-mae:1480.103938 
    [236]	train-mae:1471.448906 
    [237]	train-mae:1462.865346 
    [238]	train-mae:1454.352540 
    [239]	train-mae:1445.897382 
    [240]	train-mae:1437.482342 
    [241]	train-mae:1429.154708 
    [242]	train-mae:1420.855410 
    [243]	train-mae:1412.598366 
    [244]	train-mae:1404.380792 
    [245]	train-mae:1396.252262 
    [246]	train-mae:1388.168890 
    [247]	train-mae:1380.132316 
    [248]	train-mae:1372.154780 
    [249]	train-mae:1364.192667 
    [250]	train-mae:1356.291026 
    [251]	train-mae:1348.441833 
    [252]	train-mae:1340.635949 
    [253]	train-mae:1332.875655 
    [254]	train-mae:1325.187144 
    [255]	train-mae:1317.557941 
    [256]	train-mae:1309.967195 
    [257]	train-mae:1302.432461 
    [258]	train-mae:1294.965483 
    [259]	train-mae:1287.532728 
    [260]	train-mae:1280.140038 
    [261]	train-mae:1272.814656 
    [262]	train-mae:1265.542284 
    [263]	train-mae:1258.299958 
    [264]	train-mae:1251.106410 
    [265]	train-mae:1243.973719 
    [266]	train-mae:1236.882736 
    [267]	train-mae:1229.871947 
    [268]	train-mae:1222.871067 
    [269]	train-mae:1215.901330 
    [270]	train-mae:1209.000321 
    [271]	train-mae:1202.167158 
    [272]	train-mae:1195.373445 
    [273]	train-mae:1188.603728 
    [274]	train-mae:1181.891615 
    [275]	train-mae:1175.253895 
    [276]	train-mae:1168.626529 
    [277]	train-mae:1162.058777 
    [278]	train-mae:1155.546968 
    [279]	train-mae:1149.092292 
    [280]	train-mae:1142.636527 
    [281]	train-mae:1136.261644 
    [282]	train-mae:1129.939391 
    [283]	train-mae:1123.605701 
    [284]	train-mae:1117.364445 
    [285]	train-mae:1111.182675 
    [286]	train-mae:1105.009851 
    [287]	train-mae:1098.897762 
    [288]	train-mae:1092.836568 
    [289]	train-mae:1086.813684 
    [290]	train-mae:1080.829685 
    [291]	train-mae:1074.870721 
    [292]	train-mae:1068.963745 
    [293]	train-mae:1063.101185 
    [294]	train-mae:1057.267628 
    [295]	train-mae:1051.489428 
    [296]	train-mae:1045.733610 
    [297]	train-mae:1040.015599 
    [298]	train-mae:1034.359030 
    [299]	train-mae:1028.703030 
    [300]	train-mae:1023.109475 
    [301]	train-mae:1017.561433 
    [302]	train-mae:1012.071212 
    [303]	train-mae:1006.606972 
    [304]	train-mae:1001.168211 
    [305]	train-mae:995.762672 
    [306]	train-mae:990.401192 
    [307]	train-mae:985.086448 
    [308]	train-mae:979.795844 
    [309]	train-mae:974.511638 
    [310]	train-mae:969.302194 
    [311]	train-mae:964.121321 
    [312]	train-mae:958.947170 
    [313]	train-mae:953.838276 
    [314]	train-mae:948.764029 
    [315]	train-mae:943.728721 
    [316]	train-mae:938.734966 
    [317]	train-mae:933.768382 
    [318]	train-mae:928.844459 
    [319]	train-mae:923.909845 
    [320]	train-mae:919.059619 
    [321]	train-mae:914.188394 
    [322]	train-mae:909.411662 
    [323]	train-mae:904.647885 
    [324]	train-mae:899.925768 
    [325]	train-mae:895.226023 
    [326]	train-mae:890.565101 
    [327]	train-mae:885.939383 
    [328]	train-mae:881.326465 
    [329]	train-mae:876.750638 
    [330]	train-mae:872.210514 
    [331]	train-mae:867.721309 
    [332]	train-mae:863.242415 
    [333]	train-mae:858.798868 
    [334]	train-mae:854.380172 
    [335]	train-mae:850.016702 
    [336]	train-mae:845.653338 
    [337]	train-mae:841.342795 
    [338]	train-mae:837.072699 
    [339]	train-mae:832.822751 
    [340]	train-mae:828.587330 
    [341]	train-mae:824.446340 
    [342]	train-mae:820.291622 
    [343]	train-mae:816.172798 
    [344]	train-mae:812.089955 
    [345]	train-mae:808.026712 
    [346]	train-mae:803.987206 
    [347]	train-mae:799.976999 
    [348]	train-mae:795.966194 
    [349]	train-mae:791.989913 
    [350]	train-mae:788.064419 
    [351]	train-mae:784.177147 
    [352]	train-mae:780.314234 
    [353]	train-mae:776.471882 
    [354]	train-mae:772.667719 
    [355]	train-mae:768.904463 
    [356]	train-mae:765.131912 
    [357]	train-mae:761.397342 
    [358]	train-mae:757.715945 
    [359]	train-mae:754.045882 
    [360]	train-mae:750.410128 
    [361]	train-mae:746.815107 
    [362]	train-mae:743.226444 
    [363]	train-mae:739.671589 
    [364]	train-mae:736.136696 
    [365]	train-mae:732.639670 
    [366]	train-mae:729.185804 
    [367]	train-mae:725.727097 
    [368]	train-mae:722.323115 
    [369]	train-mae:718.914617 
    [370]	train-mae:715.540532 
    [371]	train-mae:712.184454 
    [372]	train-mae:708.849662 
    [373]	train-mae:705.567776 
    [374]	train-mae:702.280236 
    [375]	train-mae:699.023017 
    [376]	train-mae:695.785958 
    [377]	train-mae:692.548746 
    [378]	train-mae:689.328534 
    [379]	train-mae:686.154895 
    [380]	train-mae:683.009791 
    [381]	train-mae:679.909205 
    [382]	train-mae:676.815081 
    [383]	train-mae:673.775864 
    [384]	train-mae:670.714127 
    [385]	train-mae:667.704130 
    [386]	train-mae:664.701880 
    [387]	train-mae:661.696253 
    [388]	train-mae:658.757401 
    [389]	train-mae:655.821337 
    [390]	train-mae:652.900633 
    [391]	train-mae:650.012358 
    [392]	train-mae:647.129883 
    [393]	train-mae:644.280512 
    [394]	train-mae:641.456973 
    [395]	train-mae:638.639082 
    [396]	train-mae:635.877043 
    [397]	train-mae:633.115847 
    [398]	train-mae:630.381480 
    [399]	train-mae:627.635411 
    [400]	train-mae:624.936245 
    [401]	train-mae:622.213191 
    [402]	train-mae:619.590309 
    [403]	train-mae:616.947410 
    [404]	train-mae:614.325674 
    [405]	train-mae:611.725031 
    [406]	train-mae:609.140533 
    [407]	train-mae:606.581149 
    [408]	train-mae:604.041512 
    [409]	train-mae:601.510449 
    [410]	train-mae:598.999992 
    [411]	train-mae:596.515518 
    [412]	train-mae:594.026574 
    [413]	train-mae:591.566010 
    [414]	train-mae:589.117569 
    [415]	train-mae:586.702993 
    [416]	train-mae:584.328927 
    [417]	train-mae:581.960633 
    [418]	train-mae:579.617852 
    [419]	train-mae:577.270809 
    [420]	train-mae:574.957447 
    [421]	train-mae:572.655129 
    [422]	train-mae:570.374213 
    [423]	train-mae:568.127947 
    [424]	train-mae:565.871996 
    [425]	train-mae:563.649710 
    [426]	train-mae:561.432476 
    [427]	train-mae:559.258175 
    [428]	train-mae:557.113035 
    [429]	train-mae:554.949123 
    [430]	train-mae:552.824512 
    [431]	train-mae:550.708497 
    [432]	train-mae:548.616152 
    [433]	train-mae:546.522314 
    [434]	train-mae:544.442793 
    [435]	train-mae:542.382735 
    [436]	train-mae:540.336829 
    [437]	train-mae:538.314162 
    [438]	train-mae:536.309677 
    [439]	train-mae:534.314207 
    [440]	train-mae:532.358505 
    [441]	train-mae:530.386489 
    [442]	train-mae:528.421431 
    [443]	train-mae:526.504104 
    [444]	train-mae:524.611944 
    [445]	train-mae:522.726750 
    [446]	train-mae:520.860774 
    [447]	train-mae:519.022482 
    [448]	train-mae:517.175487 
    [449]	train-mae:515.314765 
    [450]	train-mae:513.518156 
    [451]	train-mae:511.722009 
    [452]	train-mae:509.942913 
    [453]	train-mae:508.177169 
    [454]	train-mae:506.445086 
    [455]	train-mae:504.711818 
    [456]	train-mae:502.997092 
    [457]	train-mae:501.300313 
    [458]	train-mae:499.616363 
    [459]	train-mae:497.937266 
    [460]	train-mae:496.283095 
    [461]	train-mae:494.649890 
    [462]	train-mae:493.007335 
    [463]	train-mae:491.398274 
    [464]	train-mae:489.779353 
    [465]	train-mae:488.197928 
    [466]	train-mae:486.620059 
    [467]	train-mae:485.076444 
    [468]	train-mae:483.533924 
    [469]	train-mae:481.996972 
    [470]	train-mae:480.465468 
    [471]	train-mae:478.950664 
    [472]	train-mae:477.452232 
    [473]	train-mae:475.982949 
    [474]	train-mae:474.503036 
    [475]	train-mae:473.051188 
    [476]	train-mae:471.628424 
    [477]	train-mae:470.201909 
    [478]	train-mae:468.784772 
    [479]	train-mae:467.358357 
    [480]	train-mae:465.950868 
    [481]	train-mae:464.592783 
    [482]	train-mae:463.239298 
    [483]	train-mae:461.870800 
    [484]	train-mae:460.526375 
    [485]	train-mae:459.185355 
    [486]	train-mae:457.848640 
    [487]	train-mae:456.519475 
    [488]	train-mae:455.244991 
    [489]	train-mae:453.967753 
    [490]	train-mae:452.696435 
    [491]	train-mae:451.418708 
    [492]	train-mae:450.166651 
    [493]	train-mae:448.931742 
    [494]	train-mae:447.692380 
    [495]	train-mae:446.486667 
    [496]	train-mae:445.269925 
    [497]	train-mae:444.082073 
    [498]	train-mae:442.910655 
    [499]	train-mae:441.713271 
    [500]	train-mae:440.555049 
    [501]	train-mae:439.415022 
    [502]	train-mae:438.270180 
    [503]	train-mae:437.131462 
    [504]	train-mae:436.006530 
    [505]	train-mae:434.899086 
    [506]	train-mae:433.813731 
    [507]	train-mae:432.737160 
    [508]	train-mae:431.669464 
    [509]	train-mae:430.605650 
    [510]	train-mae:429.525029 
    [511]	train-mae:428.480592 
    [512]	train-mae:427.449069 
    [513]	train-mae:426.418111 
    [514]	train-mae:425.377540 
    [515]	train-mae:424.384602 
    [516]	train-mae:423.382005 
    [517]	train-mae:422.400483 
    [518]	train-mae:421.411609 
    [519]	train-mae:420.437903 
    [520]	train-mae:419.473441 
    [521]	train-mae:418.533871 
    [522]	train-mae:417.590071 
    [523]	train-mae:416.674777 
    [524]	train-mae:415.766030 
    [525]	train-mae:414.846542 
    [526]	train-mae:413.936322 
    [527]	train-mae:413.066709 
    [528]	train-mae:412.180397 
    [529]	train-mae:411.316706 
    [530]	train-mae:410.448002 
    [531]	train-mae:409.601407 
    [532]	train-mae:408.742467 
    [533]	train-mae:407.904580 
    [534]	train-mae:407.065266 
    [535]	train-mae:406.244469 
    [536]	train-mae:405.434620 
    [537]	train-mae:404.614668 
    [538]	train-mae:403.814475 
    [539]	train-mae:403.001564 
    [540]	train-mae:402.211869 
    [541]	train-mae:401.436782 
    [542]	train-mae:400.672773 
    [543]	train-mae:399.909559 
    [544]	train-mae:399.166250 
    [545]	train-mae:398.441695 
    [546]	train-mae:397.694895 
    [547]	train-mae:396.964136 
    [548]	train-mae:396.242084 
    [549]	train-mae:395.522432 
    [550]	train-mae:394.817782 
    [551]	train-mae:394.107621 
    [552]	train-mae:393.412503 
    [553]	train-mae:392.727719 
    [554]	train-mae:392.033649 
    [555]	train-mae:391.361081 
    [556]	train-mae:390.699272 
    [557]	train-mae:390.030859 
    [558]	train-mae:389.384710 
    [559]	train-mae:388.757938 
    [560]	train-mae:388.129149 
    [561]	train-mae:387.499299 
    [562]	train-mae:386.886623 
    [563]	train-mae:386.271941 
    [564]	train-mae:385.675250 
    [565]	train-mae:385.073770 
    [566]	train-mae:384.481562 
    [567]	train-mae:383.887956 
    [568]	train-mae:383.302788 
    [569]	train-mae:382.726841 
    [570]	train-mae:382.176929 
    [571]	train-mae:381.625219 
    [572]	train-mae:381.074613 
    [573]	train-mae:380.530767 
    [574]	train-mae:379.976462 
    [575]	train-mae:379.431943 
    [576]	train-mae:378.907985 
    [577]	train-mae:378.383507 
    [578]	train-mae:377.873311 
    [579]	train-mae:377.344788 
    [580]	train-mae:376.826962 
    [581]	train-mae:376.328238 
    [582]	train-mae:375.839681 
    [583]	train-mae:375.347706 
    [584]	train-mae:374.865030 
    [585]	train-mae:374.376676 
    [586]	train-mae:373.888107 
    [587]	train-mae:373.403500 
    [588]	train-mae:372.939412 
    [589]	train-mae:372.473919 
    [590]	train-mae:371.994306 
    [591]	train-mae:371.525349 
    [592]	train-mae:371.086019 
    [593]	train-mae:370.645056 
    [594]	train-mae:370.207884 
    [595]	train-mae:369.773641 
    [596]	train-mae:369.338091 
    [597]	train-mae:368.913516 
    [598]	train-mae:368.502899 
    [599]	train-mae:368.084967 
    [600]	train-mae:367.670317 
    [601]	train-mae:367.263955 
    [602]	train-mae:366.847853 
    [603]	train-mae:366.436179 
    [604]	train-mae:366.048399 
    [605]	train-mae:365.645125 
    [606]	train-mae:365.253502 
    [607]	train-mae:364.870736 
    [608]	train-mae:364.500847 
    [609]	train-mae:364.133726 
    [610]	train-mae:363.771803 
    [611]	train-mae:363.410223 
    [612]	train-mae:363.050852 
    [613]	train-mae:362.696469 
    [614]	train-mae:362.339207 
    [615]	train-mae:361.984518 
    [616]	train-mae:361.649113 
    [617]	train-mae:361.312180 
    [618]	train-mae:360.994367 
    [619]	train-mae:360.673734 
    [620]	train-mae:360.348808 
    [621]	train-mae:360.029034 
    [622]	train-mae:359.704676 
    [623]	train-mae:359.395814 
    [624]	train-mae:359.072594 
    [625]	train-mae:358.751572 
    [626]	train-mae:358.448929 
    [627]	train-mae:358.153140 
    [628]	train-mae:357.860525 
    [629]	train-mae:357.562537 
    [630]	train-mae:357.275710 
    [631]	train-mae:357.000294 
    [632]	train-mae:356.719759 
    [633]	train-mae:356.444318 
    [634]	train-mae:356.154221 
    [635]	train-mae:355.886958 
    [636]	train-mae:355.598172 
    [637]	train-mae:355.324286 
    [638]	train-mae:355.064470 
    [639]	train-mae:354.800163 
    [640]	train-mae:354.540375 
    [641]	train-mae:354.289311 
    [642]	train-mae:354.028398 
    [643]	train-mae:353.772881 
    [644]	train-mae:353.520438 
    [645]	train-mae:353.276525 
    [646]	train-mae:353.024192 
    [647]	train-mae:352.777621 
    [648]	train-mae:352.542546 
    [649]	train-mae:352.303843 
    [650]	train-mae:352.059745 
    [651]	train-mae:351.826299 
    [652]	train-mae:351.602506 
    [653]	train-mae:351.373217 
    [654]	train-mae:351.173283 
    [655]	train-mae:350.961466 
    [656]	train-mae:350.733710 
    [657]	train-mae:350.523603 
    [658]	train-mae:350.304692 
    [659]	train-mae:350.095932 
    [660]	train-mae:349.902987 
    [661]	train-mae:349.707110 
    [662]	train-mae:349.505373 
    [663]	train-mae:349.300055 
    [664]	train-mae:349.113992 
    [665]	train-mae:348.911982 
    [666]	train-mae:348.723179 
    [667]	train-mae:348.529343 
    [668]	train-mae:348.337373 
    [669]	train-mae:348.154756 
    [670]	train-mae:347.960106 
    [671]	train-mae:347.778534 
    [672]	train-mae:347.604891 
    [673]	train-mae:347.429727 
    [674]	train-mae:347.251056 
    [675]	train-mae:347.072742 
    [676]	train-mae:346.899120 
    [677]	train-mae:346.728604 
    [678]	train-mae:346.556882 
    [679]	train-mae:346.385294 
    [680]	train-mae:346.227314 
    [681]	train-mae:346.078401 
    [682]	train-mae:345.917029 
    [683]	train-mae:345.758179 
    [684]	train-mae:345.597993 
    [685]	train-mae:345.447655 
    [686]	train-mae:345.291266 
    [687]	train-mae:345.150456 
    [688]	train-mae:345.000310 
    [689]	train-mae:344.851380 
    [690]	train-mae:344.707677 
    [691]	train-mae:344.569884 
    [692]	train-mae:344.426404 
    [693]	train-mae:344.285309 
    [694]	train-mae:344.147183 
    [695]	train-mae:344.012467 
    [696]	train-mae:343.869215 
    [697]	train-mae:343.730211 
    [698]	train-mae:343.590419 
    [699]	train-mae:343.452977 
    [700]	train-mae:343.321873 
    [701]	train-mae:343.208045 
    [702]	train-mae:343.079088 
    [703]	train-mae:342.961824 
    [704]	train-mae:342.832735 
    [705]	train-mae:342.708367 
    [706]	train-mae:342.572700 
    [707]	train-mae:342.443570 
    [708]	train-mae:342.321067 
    [709]	train-mae:342.190461 
    [710]	train-mae:342.078023 
    [711]	train-mae:341.953980 
    [712]	train-mae:341.837875 
    [713]	train-mae:341.718661 
    [714]	train-mae:341.606255 
    [715]	train-mae:341.489726 
    [716]	train-mae:341.373981 
    [717]	train-mae:341.274823 
    [718]	train-mae:341.167066 
    [719]	train-mae:341.063419 
    [720]	train-mae:340.961507 
    [721]	train-mae:340.845469 
    [722]	train-mae:340.743312 
    [723]	train-mae:340.645316 
    [724]	train-mae:340.531569 
    [725]	train-mae:340.441670 
    [726]	train-mae:340.340036 
    [727]	train-mae:340.241951 
    [728]	train-mae:340.145181 
    [729]	train-mae:340.044116 
    [730]	train-mae:339.951674 
    [731]	train-mae:339.856949 
    [732]	train-mae:339.756244 
    [733]	train-mae:339.660697 
    [734]	train-mae:339.562088 
    [735]	train-mae:339.469290 
    [736]	train-mae:339.367604 
    [737]	train-mae:339.277628 
    [738]	train-mae:339.197157 
    [739]	train-mae:339.111065 
    [740]	train-mae:339.025687 
    [741]	train-mae:338.930541 
    [742]	train-mae:338.841293 
    [743]	train-mae:338.768664 
    [744]	train-mae:338.689688 
    [745]	train-mae:338.617343 
    [746]	train-mae:338.542955 
    [747]	train-mae:338.467063 
    [748]	train-mae:338.396627 
    [749]	train-mae:338.314317 
    [750]	train-mae:338.225534 
    [751]	train-mae:338.142449 
    [752]	train-mae:338.060645 
    [753]	train-mae:337.982298 
    [754]	train-mae:337.916066 
    [755]	train-mae:337.843843 
    [756]	train-mae:337.765940 
    [757]	train-mae:337.686265 
    [758]	train-mae:337.615854 
    [759]	train-mae:337.540273 
    [760]	train-mae:337.473141 
    [761]	train-mae:337.393641 
    [762]	train-mae:337.322303 
    [763]	train-mae:337.256498 
    [764]	train-mae:337.187360 
    [765]	train-mae:337.120574 
    [766]	train-mae:337.048274 
    [767]	train-mae:336.976216 
    [768]	train-mae:336.909303 
    [769]	train-mae:336.843433 
    [770]	train-mae:336.779247 
    [771]	train-mae:336.719672 
    [772]	train-mae:336.642831 
    [773]	train-mae:336.569474 
    [774]	train-mae:336.504695 
    [775]	train-mae:336.439487 
    [776]	train-mae:336.379982 
    [777]	train-mae:336.327841 
    [778]	train-mae:336.261329 
    [779]	train-mae:336.210298 
    [780]	train-mae:336.155082 
    [781]	train-mae:336.094823 
    [782]	train-mae:336.037526 
    [783]	train-mae:335.985112 
    [784]	train-mae:335.941463 
    [785]	train-mae:335.893765 
    [786]	train-mae:335.834780 
    [787]	train-mae:335.783159 
    [788]	train-mae:335.728406 
    [789]	train-mae:335.687544 
    [790]	train-mae:335.632305 
    [791]	train-mae:335.572950 
    [792]	train-mae:335.519421 
    [793]	train-mae:335.459478 
    [794]	train-mae:335.408849 
    [795]	train-mae:335.363624 
    [796]	train-mae:335.305384 
    [797]	train-mae:335.261883 
    [798]	train-mae:335.220117 
    [799]	train-mae:335.170544 
    [800]	train-mae:335.122504 
    [801]	train-mae:335.078886 
    [802]	train-mae:335.022262 
    [803]	train-mae:334.975934 
    [804]	train-mae:334.928692 
    [805]	train-mae:334.878118 
    [806]	train-mae:334.827907 
    [807]	train-mae:334.790052 
    [808]	train-mae:334.749650 
    [809]	train-mae:334.700069 
    [810]	train-mae:334.653001 
    [811]	train-mae:334.606581 
    [812]	train-mae:334.563672 
    [813]	train-mae:334.520630 
    [814]	train-mae:334.484654 
    [815]	train-mae:334.446766 
    [816]	train-mae:334.405596 
    [817]	train-mae:334.360215 
    [818]	train-mae:334.318136 
    [819]	train-mae:334.267747 
    [820]	train-mae:334.224392 
    [821]	train-mae:334.176695 
    [822]	train-mae:334.128122 
    [823]	train-mae:334.085124 
    [824]	train-mae:334.054752 
    [825]	train-mae:334.005569 
    [826]	train-mae:333.960195 
    [827]	train-mae:333.911002 
    [828]	train-mae:333.858526 
    [829]	train-mae:333.820985 
    [830]	train-mae:333.782066 
    [831]	train-mae:333.737085 
    [832]	train-mae:333.706299 
    [833]	train-mae:333.668765 
    [834]	train-mae:333.616437 
    [835]	train-mae:333.575931 
    [836]	train-mae:333.536125 
    [837]	train-mae:333.501043 
    [838]	train-mae:333.460062 
    [839]	train-mae:333.425953 
    [840]	train-mae:333.383298 
    [841]	train-mae:333.341292 
    [842]	train-mae:333.300801 
    [843]	train-mae:333.252650 
    [844]	train-mae:333.213986 
    [845]	train-mae:333.175534 
    [846]	train-mae:333.140333 
    [847]	train-mae:333.104319 
    [848]	train-mae:333.066911 
    [849]	train-mae:333.032399 
    [850]	train-mae:332.992160 
    [851]	train-mae:332.960245 
    [852]	train-mae:332.933397 
    [853]	train-mae:332.897323 
    [854]	train-mae:332.875136 
    [855]	train-mae:332.846162 
    [856]	train-mae:332.791383 
    [857]	train-mae:332.747887 
    [858]	train-mae:332.714076 
    [859]	train-mae:332.688180 
    [860]	train-mae:332.652619 
    [861]	train-mae:332.617415 
    [862]	train-mae:332.591190 
    [863]	train-mae:332.562101 
    [864]	train-mae:332.528073 
    [865]	train-mae:332.505340 
    [866]	train-mae:332.473144 
    [867]	train-mae:332.436146 
    [868]	train-mae:332.407429 
    [869]	train-mae:332.388479 
    [870]	train-mae:332.358733 
    [871]	train-mae:332.335332 
    [872]	train-mae:332.302071 
    [873]	train-mae:332.279178 
    [874]	train-mae:332.245650 
    [875]	train-mae:332.226286 
    [876]	train-mae:332.199235 
    [877]	train-mae:332.174372 
    [878]	train-mae:332.136319 
    [879]	train-mae:332.110995 
    [880]	train-mae:332.080532 
    [881]	train-mae:332.054508 
    [882]	train-mae:332.018533 
    [883]	train-mae:331.993727 
    [884]	train-mae:331.968896 
    [885]	train-mae:331.941069 
    [886]	train-mae:331.912331 
    [887]	train-mae:331.876918 
    [888]	train-mae:331.840154 
    [889]	train-mae:331.808633 
    [890]	train-mae:331.783251 
    [891]	train-mae:331.744966 
    [892]	train-mae:331.725491 
    [893]	train-mae:331.694783 
    [894]	train-mae:331.666846 
    [895]	train-mae:331.630493 
    [896]	train-mae:331.595822 
    [897]	train-mae:331.576595 
    [898]	train-mae:331.552626 
    [899]	train-mae:331.532778 
    [900]	train-mae:331.509679 
    [901]	train-mae:331.475178 
    [902]	train-mae:331.444129 
    [903]	train-mae:331.414985 
    [904]	train-mae:331.385336 
    [905]	train-mae:331.357219 
    [906]	train-mae:331.326633 
    [907]	train-mae:331.318411 
    [908]	train-mae:331.301103 
    [909]	train-mae:331.263907 
    [910]	train-mae:331.235161 
    [911]	train-mae:331.217764 
    [912]	train-mae:331.192737 
    [913]	train-mae:331.163621 
    [914]	train-mae:331.138455 
    [915]	train-mae:331.120559 
    [916]	train-mae:331.099013 
    [917]	train-mae:331.088514 
    [918]	train-mae:331.072840 
    [919]	train-mae:331.051909 
    [920]	train-mae:331.032503 
    [921]	train-mae:331.016293 
    [922]	train-mae:330.999484 
    [923]	train-mae:330.989553 
    [924]	train-mae:330.974688 
    [925]	train-mae:330.950012 
    [926]	train-mae:330.926554 
    [927]	train-mae:330.908074 
    [928]	train-mae:330.899661 
    [929]	train-mae:330.872285 
    [930]	train-mae:330.849758 
    [931]	train-mae:330.832565 
    [932]	train-mae:330.817927 
    [933]	train-mae:330.795922 
    [934]	train-mae:330.778725 
    [935]	train-mae:330.758460 
    [936]	train-mae:330.733823 
    [937]	train-mae:330.733445 
    [938]	train-mae:330.717052 
    [939]	train-mae:330.688394 
    [940]	train-mae:330.660727 
    [941]	train-mae:330.646538 
    [942]	train-mae:330.626778 
    [943]	train-mae:330.611869 
    [944]	train-mae:330.603971 
    [945]	train-mae:330.579878 
    [946]	train-mae:330.558983 
    [947]	train-mae:330.555788 
    [948]	train-mae:330.531639 
    [949]	train-mae:330.507204 
    [950]	train-mae:330.478161 
    [951]	train-mae:330.460278 
    [952]	train-mae:330.432101 
    [953]	train-mae:330.404258 
    [954]	train-mae:330.389773 
    [955]	train-mae:330.367172 
    [956]	train-mae:330.358783 
    [957]	train-mae:330.341015 
    [958]	train-mae:330.318397 
    [959]	train-mae:330.299355 
    [960]	train-mae:330.294672 
    [961]	train-mae:330.281769 
    [962]	train-mae:330.258095 
    [963]	train-mae:330.221571 
    [964]	train-mae:330.202198 
    [965]	train-mae:330.180791 
    [966]	train-mae:330.166164 
    [967]	train-mae:330.140418 
    [968]	train-mae:330.129800 
    [969]	train-mae:330.113136 
    [970]	train-mae:330.099186 
    [971]	train-mae:330.086184 
    [972]	train-mae:330.066299 
    [973]	train-mae:330.038565 
    [974]	train-mae:330.024128 
    [975]	train-mae:330.020755 
    [976]	train-mae:329.995042 
    [977]	train-mae:329.987156 
    [978]	train-mae:329.985100 
    [979]	train-mae:329.962926 
    [980]	train-mae:329.927971 
    [981]	train-mae:329.919572 
    [982]	train-mae:329.901623 
    [983]	train-mae:329.882777 
    [984]	train-mae:329.861025 
    [985]	train-mae:329.844719 
    [986]	train-mae:329.831156 
    [987]	train-mae:329.823362 
    [988]	train-mae:329.800200 
    [989]	train-mae:329.781766 
    [990]	train-mae:329.771407 
    [991]	train-mae:329.766825 
    [992]	train-mae:329.740569 
    [993]	train-mae:329.724373 
    [994]	train-mae:329.711776 
    [995]	train-mae:329.697161 
    [996]	train-mae:329.675148 
    [997]	train-mae:329.661298 
    [998]	train-mae:329.642715 
    [999]	train-mae:329.620062 
    [1000]	train-mae:329.610339 
    [1001]	train-mae:329.603032 
    [1002]	train-mae:329.584522 
    [1003]	train-mae:329.571670 
    [1004]	train-mae:329.556311 
    [1005]	train-mae:329.535184 
    [1006]	train-mae:329.513760 
    [1007]	train-mae:329.507204 
    [1008]	train-mae:329.491197 
    [1009]	train-mae:329.465127 
    [1010]	train-mae:329.458609 
    [1011]	train-mae:329.431981 
    [1012]	train-mae:329.410317 
    [1013]	train-mae:329.397373 
    [1014]	train-mae:329.390913 
    [1015]	train-mae:329.376693 
    [1016]	train-mae:329.357965 
    [1017]	train-mae:329.335780 
    [1018]	train-mae:329.335230 
    [1019]	train-mae:329.334532 
    [1020]	train-mae:329.327122 
    [1021]	train-mae:329.311985 
    [1022]	train-mae:329.303988 
    [1023]	train-mae:329.290783 


Predict using xgb9.


```R
xgb9.yield = predict(xgb9, test.xgb)
range(train$yield)
range(xgb9.yield)
xgb9.guess = cbind.data.frame(test$id, xgb9.yield)
colnames(xgb9.guess) = c('id', 'yield')
write.csv(xgb9.guess, 'submission.csv', row.names = F)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>1945.53061</li><li>8969.40184</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>2469.671875</li><li>8627.3466796875</li></ol>


