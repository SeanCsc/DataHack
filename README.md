# DataHack-Yale-2018

Data Hackthon Challenge
```
ENJOY SOLVING REAL PROBLEM
```
## Problem
The goal is to help improve the real estate market. 

## File and information provided 
The Grand List of taxable property for town of fiarfield. It contains some measurement about the development of the town, like 
real estate regular net, motor vehicle net..(from 2004-2006,2009,2009-2017). To improve the development of Fairfield town, we need
to look at the development of other towns. Also, we have known 4 peer towns.

## Outstanding part
Difficulty: NO DATA.

#### Peer cities and hidden peers

Because we only have the grand list. So we need to figure out what might be useful. 

First thing is to find similar cities. Due to the time constraint, we set the geographical range to be the towns in CT.

Because our goal is to increase the performance of Fairfield by looking other cities properties, so we would filter the city with 
low income. Also, if the population or square is so small, we would also filter them.

Then, we use k-means to cluster them. Because we have known some labels. If k-means algorithm could cluster known peer towns into
one cluster, that's good. If not, we would reduce features based on the variance of features from peer cities. (If the variance is large,
which means the feature doesn't fit well for the peer towns). Luckily, when applying kmeans with k = 2, we can get one cluster containing 
all of the peer towns.

Another method we use is semi-supervised learning clustering- **label propagation**. The results got from these two method are same.

So the hidden peer cities we chose are reasonable.

#### Feature 
Income, Population, Square, Household, Averaged age...
Also, ** distance to NYC **

#### Model
Next task is to get PPE for fairfield. PPE is the expenditure for each person. It could be used to reflect what the government 
invest on the education. In this part we need to solve two problems: check which features are related to PPE. Predict PPE for 
Fairfield and compare the result. (which is used to see if the model is robust)

Use Lasso model and random forest to see the feature importance.

And compare the results of these two models.
#### Last but not least: compare the real estat market of Fairfield with other peers. Explore the relationship between educational funding and real estat market. 
**Box Plot** - Although Fairfield ranked higher, the trend is not so good.

**Considering the features correlation and use step increase**

**Give comparison to be more straightforward.** For example, 10% more education investment is similar to 20% more on safety. (which is directly related with real estate market)

#### What I learn from this challenge
Data sense: what might be useful
Model: How to avoid overfitting for small sample of data. Simple model, cross validation. 
How to do presentation vividly
