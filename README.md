# Investigating Algorithmic Stock Market Trading using Efficient Ensemble Techniques

This is an assisting repository for the published paper investigating ensemble methods in algorithmic trading. It is currently pending peer review. It was written by Khaled Sharif and Mohammad Abu-Ghazaleh, and was supervised by Dr Ramzi Saifan.

<i>Recent advances in the machine learning field have given rise to efficient ensemble methods that accurately forecast time-series. In this paper, we will use the Quantopian algorithmic stock market trading simulator to assess ensemble method performance in daily prediction and trading; simulation results show significant returns relative to the benchmark and strengthen the role of machine learning in stock market trading.</i>

<img src="http://i.imgur.com/3pIAHxp.png" />

<center> <sub> Figure 1: The graph above shows the cumulative returns of each of the three algorithms when working with 100 automatically selected stocks (selected at the start of each month) and using one classifier to predict the trading stocks. </sub>  </center>

The graph above can be produced by following the series of steps outlined by the Quantopian API. We being by initializing the simulator with both machine learning specific variables and stock trading specific variables.

```python

def initialize(context):
    set_symbol_lookup_date('2012-01-01')
    
    # Parameters to be changed
    
    context.model = ExtraTreesClassifier(n_estimators=300)
    context.lookback = 14
    context.history_range = 1000
    context.beta_coefficient = 0.0
    context.percentage_change = 0.025
    context.maximum_leverage = 2.0
    context.number_of_stocks = 150
    context.maximum_pe_ratio = 8
    context.maximum_market_cap = 0.1e9
    context.starting_probability = 0.5
    
    # End of parameters

    schedule_function(create_model, date_rules.month_start(), time_rules.market_open())
    schedule_function(rebalance, date_rules.month_start(), time_rules.market_open())
    schedule_function(trade, date_rules.every_day(), time_rules.market_open())

    context.algorithm_returns = []
    context.longs = []
    context.shorts = []
    context.training_stocks = symbols('SPY')
    context.trading_stocks  = []
    context.beta = 1.0
    context.beta_list = []
    context.completed = False

```

Following the variable initialization, we define two functions to run monthly and daily, respectively. The first function is the model creation, and it is outlined below. It runs once every month, on the first trading day of that month, and uses the last 1000 days (or whatever is defined by the global "history range" variable, as training data for the machine learning algorithm.


```python

def create_model(context, data):
    X = []
    Y = [] 
    
    for S in context.training_stocks:
        recent_prices = history(context.history_range, '1d', 'price')[S].values
        recent_lows   = history(context.history_range, '1d', 'low')[S].values
        recent_highs  = history(context.history_range, '1d', 'high')[S].values
        recent_closes = history(context.history_range, '1d', 'close_price')[S].values

        atr = talib.ATR(recent_highs, recent_lows, recent_closes, timeperiod=14)
        prev_close = np.roll(recent_closes, 2)
        upside_signal = (recent_prices - (prev_close + atr)).tolist()
        downside_signal = (prev_close - (recent_prices + atr)).tolist()
        price_changes = np.diff(recent_prices).tolist()
        upper, middle, lower = talib.BBANDS(recent_prices,timeperiod=10,nbdevup=2,nbdevdn=2,matype=1)
        upper = upper.tolist()
        middle = middle.tolist()
        lower = lower.tolist()
   
        for i in range(15, context.history_range-context.lookback-1):
            Z = price_changes[i:i+context.lookback] + \
                upside_signal[i:i+context.lookback] + \
                downside_signal[i:i+context.lookback] + \
                upper[i:i+context.lookback] + \
                middle[i:i+context.lookback] + \
                lower[i:i+context.lookback] 
                
            if (np.any(np.isnan(Z)) or not np.all(np.isfinite(Z))): continue

            X.append(Z)

            if abs(price_changes[i+context.lookback]) > abs(price_changes[i]*(1+context.percentage_change)):
                if price_changes[i+context.lookback] > 0:
                    Y.append(+1)
                else:
                    Y.append(-1)
            else:
                Y.append(0)

    context.model.fit(X, Y) 
    

```


Automatic stock selection is also done monthly, at the start of every month, and uses basic fundamental analysis to automatically select 100 stocks from the NYSE and NASDAQ stock markets in a way free of survivorship bias. 


```python

def before_trading_start(context): 
    if context.completed: return

    fundamental_df = get_fundamentals(query(fundamentals.valuation.market_cap)
        .filter(fundamentals.company_reference.primary_exchange_id == 'NAS' or 
                fundamentals.company_reference.primary_exchange_id == 'NYSE')
        .filter(fundamentals.valuation_ratios.pe_ratio < context.maximum_pe_ratio)
        .filter(fundamentals.valuation.market_cap < context.maximum_market_cap)
        .order_by(fundamentals.valuation.market_cap.desc())
        .limit(context.number_of_stocks)) 
    update_universe(fundamental_df.columns.values)
    
    context.trading_stocks = [stock for stock in fundamental_df]
    context.completed = True

```


The final part of the simulator is the actual day-to-day trading that occurs in the simulator at the start of every trading day. When the one classifier method is used, the trading algorithm only takes action when the classifier is confident in its classification over a certain threshold (this was fixed to 60%). When the two classifier method is used, the trading algorithm only takes action when the two classifiers agree on a classification. For both methods, inaction meant that the current portfolio remained unchanged.


```python

def trade(context, data):
    if (context.account.leverage > context.maximum_leverage): return
    
    if not context.model: return
    
    for stock in context.trading_stocks: 
        if stock not in data: 
            context.trading_stocks.remove(stock)
            
    for stock in context.trading_stocks:  
        if stock.security_end_date < get_datetime(): 
            context.trading_stocks.remove(stock)
        if stock in security_lists.leveraged_etf_list: 
            context.trading_stocks.remove(stock)
    
    for one_stock in context.trading_stocks:
        if get_open_orders(one_stock): continue

        recent_prices = history(context.lookback+30, '1d', 'price')[one_stock].values
        recent_lows   = history(context.lookback+30, '1d', 'low')[one_stock].values
        recent_highs  = history(context.lookback+30, '1d', 'high')[one_stock].values
        recent_closes = history(context.lookback+30, '1d', 'close_price')[one_stock].values
        
        if (np.any(np.isnan(recent_prices)) or not np.all(np.isfinite(recent_prices))): continue
        if (np.any(np.isnan(recent_lows)) or not np.all(np.isfinite(recent_lows))): continue
        if (np.any(np.isnan(recent_highs)) or not np.all(np.isfinite(recent_highs))): continue
        if (np.any(np.isnan(recent_closes)) or not np.all(np.isfinite(recent_closes))): continue
            
        atr = talib.ATR(recent_highs, recent_lows, recent_closes, timeperiod=14)
        prev_close = np.roll(recent_closes, 2)
        upside_signal = (recent_prices - (prev_close + atr)).tolist()
        downside_signal = (prev_close - (recent_prices + atr)).tolist()
        price_changes = np.diff(recent_prices).tolist()
        upper, middle, lower = talib.BBANDS(recent_prices,timeperiod=10,nbdevup=2,nbdevdn=2,matype=1)
        upper = upper.tolist()
        middle = middle.tolist()
        lower = lower.tolist()
        
        L = context.lookback        
        Z = price_changes[-L:] + upside_signal[-L:] + downside_signal[-L:] + \
            upper[-L:] + middle[-L:] + lower[-L:] 
            
        if (np.any(np.isnan(Z)) or not np.all(np.isfinite(Z))): continue
            
        prediction = context.model.predict(Z)
        predict_proba = context.model.predict_proba(Z)
        probability = predict_proba[0][prediction+1]
        
        p_desired = context.starting_probability + 0.1*context.portfolio.returns   
        
        if probability > p_desired:
            if prediction > 0:
                if one_stock in context.shorts:
                    order_target_percent(one_stock, 0)
                    context.shorts.remove(one_stock)
                elif not one_stock in context.longs:
                    context.longs.append(one_stock)
                    
            elif prediction < 0:
                if one_stock in context.longs:
                    order_target_percent(one_stock, 0)
                    context.longs.remove(one_stock)
                elif not one_stock in context.shorts:
                    context.shorts.append(one_stock)
                    
            else:
                order_target_percent(one_stock, 0)
                if one_stock in context.longs:    context.longs.remove(one_stock)
                elif one_stock in context.shorts: context.shorts.remove(one_stock)
        
                
    if get_open_orders(): return
    
    for one_stock in context.longs:
        if not one_stock in context.trading_stocks:
            context.longs.remove(one_stock)
        else:
            order_target_percent(one_stock, \
               context.maximum_leverage/(len(context.longs)+len(context.shorts)))
    
    for one_stock in context.shorts:
        if not one_stock in context.trading_stocks:
            context.shorts.remove(one_stock)
        else:
            order_target_percent(one_stock, \
              (-1.0)*context.maximum_leverage/(len(context.longs)+len(context.shorts)))
        
    order_target_percent(symbol('SPY'),  \
       (-1.0)*context.maximum_leverage*(context.beta*context.beta_coefficient))


```



<center> <sub> Table 1: The table below compares the average values of the alpha and beta coefficients over 12-month periods for each of the three classification methods when used in simulation over the time-period 2010 to 2015. </sub> </center>



|                                       |     12-month Alpha    |     12-month Alpha     |     12-month Beta     |      12-month Beta     |
|:-------------------------------------:|:---------------------:|:----------------------:|:---------------------:|:----------------------:|
|                                       | One Classifier Method | Two Classifiers Method | One Classifier Method | Two Classifiers Method |
|        Random Forest Classifier       |          0.40         |          1.29          |          1.89         |          2.79          |
| Extremely Randomized Trees Classifier |          0.40         |          1.05          |          1.25         |          2.77          |
|      Gradient Boosting Classifier     |          0.62         |          1.37          |          1.74         |          4.70          |


<center> <sub> Table 2: The table below compares the average values of the Sharpe, Sortino and Information ratios over 12-month periods for each of the three classification methods when used in simulation over the time-period 2010 to 2015. </sub> </center>


|                                       |      Sharpe Ratio     |      Sharpe Ratio      |     Sortino Ratio     |      Sortino Ratio     |   Information Ratio   |    Information Ratio   |
|:-------------------------------------:|:---------------------:|:----------------------:|:---------------------:|:----------------------:|:---------------------:|:----------------------:|
|                                       | One Classifier Method | Two Classifiers Method | One Classifier Method | Two Classifiers Method | One Classifier Method | Two Classifiers Method |
|        Random Forest Classifier       |          2.26         |          3.42          |          4.06         |          5.28          |          0.11         |          0.10          |
| Extremely Randomized Trees Classifier |          2.68         |          3.24          |          4.07         |          3.25          |          0.10         |          0.10          |
|      Gradient Boosting Classifier     |          3.61         |          3.84          |          5.28         |          5.73          |          0.15         |          0.13          |


<center> <sub> Table 3: The table below compares the average values of the volatility and maximum draw-down indicators over 12-month periods for each of the three classification methods when used in simulation over the time-period 2010 to 2015. </sub> </center>


|                                       |       Volatility      |       Volatility       |   Maximum Draw-down   |    Maximum Draw-down   |
|:-------------------------------------:|:---------------------:|:----------------------:|:---------------------:|:----------------------:|
|                                       | One Classifier Method | Two Classifiers Method | One Classifier Method | Two Classifiers Method |
|        Random Forest Classifier       |          0.24         |          0.35          |         11.55%        |         21.45%         |
| Extremely Randomized Trees Classifier |          0.23         |          0.49          |         11.69%        |         25.25%         |
|      Gradient Boosting Classifier     |          0.22         |          0.38          |         24.00%        |         24.02%         |
