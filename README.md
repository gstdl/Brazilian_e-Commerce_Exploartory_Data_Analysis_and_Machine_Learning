# Brazilian e-Commerce Exploartory Data Analysis and Machine_Learning
This Repository is made to store data about my final project as a requirement to graduate from Data Science Job Connector Program at Purwadhika Startup &amp; Coding School

## A Weird Final Project
This final project contains a quite comprehensive Exploratory Data Analysis (EDA) but nearly none of the EDA is involved in the machine learning process. 

**Weird isn't it?**

Well, the plan A was to build an item, customer, and seller clustering model but my computer doesn't have the computational power to live up to this challenge.

Before I dig deeper, allow me to briefly explain about the data used in this project.

The dataset used in this final project is obtained a market place named [Olist](http://www.olist.com/) which is located in Brazil.

It contains of more than 100,000 orders from various marketplaces between 2016 to 2018. 

This dataset is seperated into 9 datasets which follows the merging schema below.

![](https://i.imgur.com/HRhd2Y0.png "Olist Dataset Merging Schema")

After merging the dataset consist of 118315 rows and  39 columns. [Click here to check the dataset on Kaggle.](https://www.kaggle.com/olistbr/brazilian-ecommerce)

**Wait? That doesn't look like a lot of data**

Yeah, if you look at the number of rows and columns, it looks like a small dataset. 

However, datasets is more than row and column quantity. 

You should consider the data cleaning process, understanding every feature in the dataset, as well as exploring every interisting unique value in each feature.

I needed to write 3 clean notebooks only to explore this dataset and I feel like there is still more to explore from this dataset.

I won't be explaining my EDA findings in this readme. Feel free to read the notebooks uploaded in this repository.

You can also try using your own code to explore the dataset further in each notebooks by using Google Colab and your own Kaggle API key 
(To obtain a Kaggle API key, sign Up for a Kaggle if you haven't, from your Kaggle account go to your account setting page and click on `Create New API Token`)

Back to the initial idea, after EDA, I decided to start developing my first model to cluster the items. The features I selected was product_category_name, product_weight, product_length, product_height, product_width, product_volume, price, and seller_city.
After getting dummies out the features, I ended up with 689 features. This takes so many computational power and ended up burdening my computer.
Next, I tried reducing the 689 feature dimension by using Principal Component Analysis (PCA) but ended up with less than 1% explained variance ratio (for n_components = 3)

Considering how much computation power needed to complete the initial idea, I decided to change the idea to another idea.

**Changing Ideas**

During EDA, I discovered that there was lots of missing values in product_category_name feature with other features information available.
But, I decided to postpone this idea for when I have enough computational power to finish this challenge. 
In the end, I decided to try developing a computer simulation application which can help in deciding on where should an item warehouse be placed.
However, in order to evaluate the model, I focused on sample density in each clusters and euclidian distance between particular points in each cluster.

With that being said, to evaluate the model I decided to use the Traveling Salesman Problem (TSP) to decide the minimum distance a person should be traveling in each cluster when picking up orders and delivering orders.

## Limitations

There are several limitations considered in this final project which are as follows:

1. The TSP has a limit on how many points it can visit before stopping. The set of stopping points are determined randomly without using permutation (The TSP model is not optimized to have global minimum distance)
2. The TSP is not considered to start from the cluster center nor end in the cluster center. When there is only 1 point the set of stopping points, the model will move 0 distance.

## Future Plans for This Dataset

1. Completing the item, customer, seller clustering.
2. Develop a classification model to fill in the missing product_category_name feature.
3. Develop a time-series model or regression model to predict future new customers from the dataset.
