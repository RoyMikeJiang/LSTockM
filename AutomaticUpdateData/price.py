import pymysql
import sys
import yfinance as yf

## Database Config
host = 'localhost'
user = 'root'
password = 'password'
database = 'lstockm'

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
    ticker = yf.Ticker(stock_info[1])
    market_data = ticker.history(period="1d", auto_adjust=False)
    date = ''
    if(len(market_data)==0):
        print("Error when updating the stock(_id) : {}".format(stock_info[0]))
        continue
    else:
        date = market_data.index[0].strftime('%Y-%m-%d')
        market_data = market_data.iloc[0]
    sql = ''
    try:
        sql = "INSERT INTO ohlcv (stock_id, data_source_id, price_date, open_price, high_price, low_price, close_price, adj_close_price, volume) VALUES ({}, 1, '{}', {}, {}, {}, {}, {}, {})".format(stock_info[0], date, market_data['Open'], market_data['High'], market_data['Low'], market_data['Close'], market_data['Adj Close'], market_data['Volume'])
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        db.rollback()
        print("Error when updating the stock(_id): {}\n SQL statement: {}\n Error info: {}".format(stock_info[0], sql, e))
        continue

db.close()