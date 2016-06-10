# Investigating Algorithmic Stock Market Trading using Efficient Ensemble Techniques

This is an assisting repository for the published paper investigating ensemble methods in algorithmic trading. It is currently pending peer review. It was written by Khaled Sharif and Mohammad Abu-Ghazaleh, and was supervised by Dr Ramzi Saifan.

<i>Recent advances in the machine learning field have given rise to efficient ensemble methods that accurately forecast time-series. In this paper, we will use the Quantopian algorithmic stock market trading simulator to assess ensemble method performance in daily prediction and trading; simulation results show significant returns relative to the benchmark and strengthen the role of machine learning in stock market trading.</i>

<img src="http://i.imgur.com/3pIAHxp.png" />

Figure 1: The graph above shows the cumulative returns of each of the three algorithms when working with 100 automatically selected stocks (selected at the start of each month) and using one classifier to predict the trading stocks.


Table 1: The table below compares the average values of the alpha and beta coefficients over 12-month periods for each of the three classification methods when used in simulation over the time-period 2010 to 2015.



|                                       |     12-month Alpha    |     12-month Alpha     |     12-month Beta     |      12-month Beta     |
|:-------------------------------------:|:---------------------:|:----------------------:|:---------------------:|:----------------------:|
|                                       | One Classifier Method | Two Classifiers Method | One Classifier Method | Two Classifiers Method |
|        Random Forest Classifier       |          0.40         |          1.29          |          1.89         |          2.79          |
| Extremely Randomized Trees Classifier |          0.40         |          1.05          |          1.25         |          2.77          |
|      Gradient Boosting Classifier     |          0.62         |          1.37          |          1.74         |          4.70          |


Table 2: The table below compares the average values of the Sharpe, Sortino and Information ratios over 12-month periods for each of the three classification methods when used in simulation over the time-period 2010 to 2015.


|                                       |      Sharpe Ratio     |      Sharpe Ratio      |     Sortino Ratio     |      Sortino Ratio     |   Information Ratio   |    Information Ratio   |
|:-------------------------------------:|:---------------------:|:----------------------:|:---------------------:|:----------------------:|:---------------------:|:----------------------:|
|                                       | One Classifier Method | Two Classifiers Method | One Classifier Method | Two Classifiers Method | One Classifier Method | Two Classifiers Method |
|        Random Forest Classifier       |          2.26         |          3.42          |          4.06         |          5.28          |          0.11         |          0.10          |
| Extremely Randomized Trees Classifier |          2.68         |          3.24          |          4.07         |          3.25          |          0.10         |          0.10          |
|      Gradient Boosting Classifier     |          3.61         |          3.84          |          5.28         |          5.73          |          0.15         |          0.13          |


Table 3: The table below compares the average values of the volatility and maximum draw-down indicators over 12-month periods for each of the three classification methods when used in simulation over the time-period 2010 to 2015.


|                                       |       Volatility      |       Volatility       |   Maximum Draw-down   |    Maximum Draw-down   |
|:-------------------------------------:|:---------------------:|:----------------------:|:---------------------:|:----------------------:|
|                                       | One Classifier Method | Two Classifiers Method | One Classifier Method | Two Classifiers Method |
|        Random Forest Classifier       |          0.24         |          0.35          |         11.55%        |         21.45%         |
| Extremely Randomized Trees Classifier |          0.23         |          0.49          |         11.69%        |         25.25%         |
|      Gradient Boosting Classifier     |          0.22         |          0.38          |         24.00%        |         24.02%         |
