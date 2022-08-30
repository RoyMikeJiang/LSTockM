import pymysql
import sys
import yfinance as yf
import time
import scipy.stats as stats
import math
import pandas as pd

def cal_VaR(days, confidence, stock_data):
    share_price = stock_data['Close'][-1]
    Z_value = stats.norm.ppf(abs(1 - confidence))
    mean_return_rate = stock_data['Close'].pct_change().mean()
    std_return_rate = stock_data['Close'].pct_change().std()
    
    rel_VaR = math.sqrt(days) * share_price * Z_value * std_return_rate 
    return rel_VaR

## Database Config
host = 'localhost'
user = 'root'
password = 'password'
database = 'lstockm'

keys = ['currentPrice','ebitdaMargins','operatingMargins','returnOnAssets','returnOnEquity','pegRatio','priceToBook','volume','bid','bidSize','ask','askSize','beta','quickRatio']
def getCurDataFromYahoo(stock_symbol: str):
    ticker = yf.Ticker(stock_symbol)
    # ticker._get_fundamentals(proxy="http://127.0.0.1:7890")
    ticker._get_fundamentals()
    market_data = ticker.info
    output_data = {}
    for key in keys:
        output_data[key] = market_data[key]
    return output_data

## Database Connection Init
db = pymysql.connect(
    host=host,
    user=user,
    password=password,
    database=database
)
cursor = db.cursor()

## Get the Stocks List
stock_list = ''
try:
    cursor.execute("SELECT stock_id, stock_symbol FROM stock_info;")
    stock_list = cursor.fetchall()
except Exception as e:
    print(e)
    db.close()
    sys.exit(1)



## Get the data from the yahoo finance
for stock_info in stock_list:
    data = getCurDataFromYahoo(stock_info[1])
    date = ''
    if(len(data)==0):
        print("Error when updating the stock(_id) : {}".format(stock_info[0]))
        continue
    
    sql = "SELECT price_date, close_price FROM ohlcv WHERE stock_id = {} ORDER BY price_date DESC LIMIT 300".format(stock_info[0])
    results_array = []
    try:
        cursor.execute(sql)
        results_array = cursor.fetchall()
    except pymysql.MySQLError as e:
        print("Error {} for execute sql: {}".format(e, sql))
        db.close()
    df = pd.DataFrame(results_array, columns = ["Date","Close"])
    df.set_index(["Date"], inplace=True)
    var = cal_VaR(5, 0.99 , df)
    sql = 'UPDATE current_data SET data_time = now() '

    for key in keys:
        sql += ', {} = {} '.format(key, data[key] if data[key] != None else 'NULL')
    sql += ', valueAtRisk = {} WHERE stock_id = {}'.format(var, stock_info[0])
    try:
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        db.rollback()
        print("Error when updating the stock(_id): {}\n SQL statement: {}\n Error info: {}".format(stock_info[0], sql, e))
        continue
    time.sleep(10)
db.close()