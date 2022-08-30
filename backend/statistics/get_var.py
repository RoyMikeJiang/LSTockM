from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import Optional
import pymysql
from utils.StaticConfig import MYSQLStaticConfig
from utils.sql_fetch_json import sql_fetch_json
import scipy.stats as stats
import math
import pandas as pd

router = APIRouter(
    prefix = '/statistics/var'
)

def cal_VaR(days, confidence, stock_data):
    share_price = stock_data['Close'][-1]
    Z_value = stats.norm.ppf(abs(1 - confidence))
    mean_return_rate = stock_data['Close'].pct_change().mean()
    std_return_rate = stock_data['Close'].pct_change().std()
    
    rel_VaR = math.sqrt(days) * share_price * Z_value * std_return_rate 
    return rel_VaR

@router.get('/')
async def get_var(stock_id : int):
    db = pymysql.connect(host=MYSQLStaticConfig.IP,port=MYSQLStaticConfig.PORT,user=MYSQLStaticConfig.USER,password=MYSQLStaticConfig.PASSWORD,database=MYSQLStaticConfig.DATABASE)
    cursor = db.cursor()
    sql = "SELECT price_date, close_price FROM ohlcv WHERE stock_id = {} ORDER BY price_date DESC LIMIT 300".format(stock_id)
    results_array = []
    try:
        cursor.execute(sql)
        results_array = cursor.fetchall()
    except pymysql.MySQLError as e:
        print("Error {} for execute sql: {}".format(e, sql))
        db.close()
        return JSONResponse(status_code=500, content={"result": "Fail", "INFO" : "Error With Database Server"})
    df = pd.DataFrame(results_array, columns = ["Date","Close"])
    df.set_index(["Date"], inplace=True)
    results_json = {"result": "Success", "var": cal_VaR(5, 0.99 , df)}
    db.close()
    return results_json