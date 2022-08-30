import pymysql
import sys
import yfinance as yf
import time

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
    cursor.execute("SELECT stock_id, stock_symbol, latest_rcmd_time FROM stock_info si LEFT OUTER JOIN (SELECT stock_id, MAX(rcmd_time) AS latest_rcmd_time FROM recommendations r GROUP BY stock_id) lr USING(stock_id) WHERE stock_id > 14;")
    stock_list = cursor.fetchall()
except Exception as e:
    print(e)
    db.close()
    sys.exit(1)

## Get the data from the yahoo finance
for stock_info in stock_list:
    ticker = yf.Ticker(stock_info[1])
    rcmd_data = ticker.recommendations
    date = ''
    if(len(rcmd_data)==0):
        print("Error when updating the stock(_id) : {}".format(stock_info[0]))
        continue
    if(stock_info[2]==None):
        pass
    elif(rcmd_data.index[-1]<=stock_info[2]):
        continue
    else:
        rcmd_data = rcmd_data[(rcmd_data.index > stock_info[2])]
    print(rcmd_data)
    sql = ''
    for index, row in rcmd_data.iterrows():
        try:
            sql = "INSERT INTO recommendations (stock_id, rcmd_time, firm, to_grade, from_grade, `action`) VALUES ({}, '{}', '{}', '{}', '{}', '{}')".format(stock_info[0], index, row['Firm'], row['To Grade'], row['From Grade'], row['Action'])
            cursor.execute(sql)
            db.commit()
        except Exception as e:
            db.rollback()
            print("Error when importing the stock(_id): {}\n SQL statement: {}\n Error info: {}".format(stock_info[0], sql, e))
            continue
    time.sleep(30)

db.close()