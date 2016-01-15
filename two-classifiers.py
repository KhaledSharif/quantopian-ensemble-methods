import talib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
import numpy as np
import pandas

def initialize(context):
    set_symbol_lookup_date('2010-01-01')
    
    # Parameters to be changed
    
    context.model1 = RandomForestClassifier(n_estimators=300, 
                                     max_depth=6, max_features=None)
    context.model2 = RandomForestClassifier(n_estimators=300, 
                                     max_depth=6, max_features=None)
    context.lookback = 14
    context.history_range = 1000
    context.beta_coefficient = 0.0
    context.percentage_change = 0.034
    context.maximum_leverage = 2.0
    context.number_of_stocks = 150
    context.maximum_pe_ratio = 8
    context.maximum_market_cap = 0.1e9
    
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
            Z = price_changes[i:i+context.lookback] + upside_signal[i:i+context.lookback] + downside_signal[i:i+context.lookback] +\
                upper[i:i+context.lookback] + middle[i:i+context.lookback] + lower[i:i+context.lookback] 
                
            if (np.any(np.isnan(Z)) or not np.all(np.isfinite(Z))): continue

            X.append(Z)

            if abs(price_changes[i+context.lookback]) > abs(price_changes[i]*(1+context.percentage_change)):
                if price_changes[i+context.lookback] > 0:
                    Y.append(+1)
                else:
                    Y.append(-1)
            else:
                Y.append(0)

    context.model1.fit(X, Y) 
    context.model2.fit(X, Y) 
    

def rebalance(context, data):
    context.completed = False
        
def trade(context, data):
    if (context.account.leverage > context.maximum_leverage): return
    
    if not context.model1: return
    
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
        Z = price_changes[-L:] + upside_signal[-L:] + downside_signal[-L:] + upper[-L:] + middle[-L:] + lower[-L:] 
            
        if (np.any(np.isnan(Z)) or not np.all(np.isfinite(Z))): continue
            
        prediction1 = context.model1.predict(Z)
        prediction2 = context.model2.predict(Z)
        
        if prediction1 == prediction2:
            if prediction1 > 0:
                if one_stock in context.shorts:
                    order_target_percent(one_stock, 0)
                    context.shorts.remove(one_stock)
                elif not one_stock in context.longs:
                    context.longs.append(one_stock)
                    
            elif prediction1 < 0:
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
            order_target_percent(one_stock, context.maximum_leverage/(len(context.longs)+len(context.shorts)))
    
    for one_stock in context.shorts:
        if not one_stock in context.trading_stocks:
            context.shorts.remove(one_stock)
        else:
            order_target_percent(one_stock, (-1.0)*context.maximum_leverage/(len(context.longs)+len(context.shorts)))
        
    order_target_percent(symbol('SPY'), (-1.0)*context.maximum_leverage*(context.beta*context.beta_coefficient))
    
  
def estimateBeta(priceY,priceX):  
    algorithm_returns = priceY
    benchmark_returns = (priceX/np.roll(priceX,1)-1).dropna().values  
    if len(algorithm_returns) <> len(benchmark_returns):  
        minlen = min(len(algorithm_returns), len(benchmark_returns))  
        if minlen > 2:  
            algorithm_returns = algorithm_returns[-minlen:]  
            benchmark_returns = benchmark_returns[-minlen:]  
        else:  
            return 1.00  
    returns_matrix = np.vstack([algorithm_returns, benchmark_returns])  
    C = np.cov(returns_matrix, ddof=1)  
    algorithm_covariance = C[0][1]  
    benchmark_variance = C[1][1]  
    beta = algorithm_covariance / benchmark_variance

    return beta

 
def handle_data(context, data):
    record(cash = context.portfolio.cash/(1000000))  
    record(lev  = context.account.leverage)
    
    context.algorithm_returns.append(context.portfolio.returns)
    if len(context.algorithm_returns) > 30:
        recent_prices = history(len(context.algorithm_returns), '1d', 'price')[symbol('SPY')]
        
        context.beta_list.append(estimateBeta(pandas.Series(context.algorithm_returns[-30:]), recent_prices))
        if len(context.beta_list) > 7: context.beta_list.pop(0)
        context.beta = np.mean(context.beta_list)
    record(Beta=context.beta)
    
    
