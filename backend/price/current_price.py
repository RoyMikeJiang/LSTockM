from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import Optional
import pymysql
from utils.StaticConfig import MYSQLStaticConfig
from utils.sql_fetch_json import sql_fetch_json
import yfinance as yf
from datetime import datetime

router = APIRouter(
    prefix = '/price/current'
)
keys = ['currentPrice','ebitdaMargins','operatingMargins','returnOnAssets','returnOnEquity','pegRatio','priceToBook','volume','bid','bidSize','ask','askSize', 'beta', 'quickRatio']

def getCurDataFromYahoo(stock_symbol: str):
    ticker = yf.Ticker(stock_symbol)
    ticker._get_fundamentals(proxy="http://127.0.0.1:7890")
    # ticker._get_fundamentals()
    market_data = ticker.info
    output_data = {}
    for key in keys:
        output_data[key] = market_data[key]
    return output_data
    
@router.get('/')
async def current_price(stock_id: int, stock_symbol: str):
    db = pymysql.connect(host=MYSQLStaticConfig.IP,port=MYSQLStaticConfig.PORT,user=MYSQLStaticConfig.USER,password=MYSQLStaticConfig.PASSWORD,database=MYSQLStaticConfig.DATABASE)
    cursor = db.cursor()
    sql = "SELECT TIMESTAMPDIFF(SECOND, data_time, now()) as passed_time, currentPrice, data_time FROM current_data WHERE stock_id = {}".format(stock_id)
    results_array = []
    try:
        cursor.execute(sql)
        results_array = await sql_fetch_json(cursor)
    except pymysql.MySQLError as e:
        print("Error {} for execute sql: {}".format(e, sql))
        db.close()
        return JSONResponse(status_code=500, content={"result": "Fail", "INFO" : "Error With Database Server"})
    if len(results_array) > 0:
        if(results_array[0]['passed_time'] < 1800):
            results_json = {"result": "Success", "data": results_array[0]['currentPrice'], "dataTime": results_array[0]['data_time']}
            db.close()
            return results_json
    else:
        return JSONResponse(status_code=500, content={"result": "Fail", "INFO" : "Error With stock_id {}".format(stock_id)})
    
    data = getCurDataFromYahoo(stock_symbol)
    # sql = "UPDATE current_data (data_time, {}) VALUES(now(), {})".format(",".join(keys),",".join(data))
    sql = 'UPDATE current_data SET data_time = now() '
    for key in keys:
        sql += ', {} = {} '.format(key, data[key] if data[key] != None else 'NULL')
    sql += ' WHERE stock_id = {}'.format(stock_id)
    try:
        cursor.execute(sql)
        db.commit()
    except pymysql.MySQLError as e:
        print("Error {} for execute sql: {}".format(e, sql))
        db.rollback()
        db.close()
        return JSONResponse(status_code=500, content={"result": "Fail", "INFO" : "Error With Database Server"})
    results_json = {"result": "Success", "data": data['currentPrice'], "dataTime": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}
    db.close()
    return results_json