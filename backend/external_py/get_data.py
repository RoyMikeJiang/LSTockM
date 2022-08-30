import pymysql
import sys
import yfinance as yf
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--stock_symbol')

if __name__ == '__main__':
    args = parser.parse_args()

    ticker = yf.Ticker(args.stock_symbol)
    ticker._get_fundamentals(proxy="http://127.0.0.1:7890")
    # ticker._get_fundamentals(proxy=proxy)

    if('symbol' not in ticker.info.keys()):
        print("No Stock Founded for Symbol: {}".format(args.stock_symbol))
        sys.exit(1)

    ## Database Config
    # host = 'localhost'
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

    sql = "SELECT exchange_id FROM exchange_info WHERE exchange_name = '{}'".format(ticker.info['exchange'])
    cursor.execute(sql)
    exchange_id = cursor.fetchall()
    if len(exchange_id)==0:
        sql = "INSERT INTO exchange_info (exchange_name, currency) VALUES ('{}','{}')".format(ticker.info['exchange'], ticker.info['currency'])
        try:
            cursor.execute(sql)
            exchange_id = cursor.lastrowid
            db.commit()
        except Exception as e:
            print("Error with inserting the exchange info. sql: {} ticker_symbol: {}".format(sql, ticker.info['symbol']))
            db.rollback()
            db.close()
            sys.exit(1)
    else:
        exchange_id = exchange_id[0][0]
    
    sql = "INSERT INTO stock_info (stock_symbol, company_name, exchange_id) VALUES ('{}','{}',{})".format(ticker.info['symbol'], ticker.info['shortName'], exchange_id)
    stock_id = 0
    try:
        cursor.execute(sql)
        stock_id = cursor.lastrowid
        db.commit()
    except Exception as e:
        print("Error with inserting the stock info. sql: {} ticker_symbol: {}".format(sql, ticker.info['symbol']))
        db.rollback()
        db.close()
        sys.exit(1)

    sql = "INSERT INTO current_data (stock_id) VALUES ({})".format(stock_id)
    try:
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        print("Error when init the current_data. stock_id: {}".format(stock_id))
        db.rollback()
        db.close()
        sys.exit(1)

    sql = "INSERT INTO predict_price (stock_id, predict_price) VALUES ({}, 1)".format(stock_id)
    try:
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        print("Error when init the predict_price. stock_id: {}".format(stock_id))
        db.rollback()
        db.close()
        sys.exit(1)
    
    market_data = ticker.history(period="10y", auto_adjust=False, proxy="http://127.0.0.1:7890")
    if(len(market_data)==0):
        print("Error with getting price. ticker_id: {}".format(stock_id))
    sql = ''
    for index, row in market_data.iterrows():
        try:
            sql = "INSERT INTO ohlcv (stock_id, data_source_id, price_date, open_price, high_price, low_price, close_price, adj_close_price, volume) VALUES ({}, 1, '{}', {}, {}, {}, {}, {}, {})".format(stock_id, index, row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'], row['Volume'])
            cursor.execute(sql)
            db.commit()
        except Exception as e:
            db.rollback()
            print("Error when updating the stock(_id): {}\n SQL statement: {}\n Error info: {}".format(stock_id, sql, e))
            continue
    db.close()