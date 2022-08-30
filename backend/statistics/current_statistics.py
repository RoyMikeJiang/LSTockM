from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import Optional
import pymysql
from utils.StaticConfig import MYSQLStaticConfig
from utils.sql_fetch_json import sql_fetch_json
import yfinance as yf
from datetime import datetime

router = APIRouter(
    prefix = '/statistics/current'
)
    
@router.get('/')
async def current_price(stock_id: int):
    db = pymysql.connect(host=MYSQLStaticConfig.IP,port=MYSQLStaticConfig.PORT,user=MYSQLStaticConfig.USER,password=MYSQLStaticConfig.PASSWORD,database=MYSQLStaticConfig.DATABASE)
    cursor = db.cursor()
    sql = "SELECT * FROM current_data WHERE stock_id = {}".format(stock_id)
    results_array = []
    try:
        cursor.execute(sql)
        results_array = await sql_fetch_json(cursor)
    except pymysql.MySQLError as e:
        print("Error {} for execute sql: {}".format(e, sql))
        db.close()
        return JSONResponse(status_code=500, content={"result": "Fail", "INFO" : "Error With Database Server"})
    if len(results_array) > 0:
        results_json = {"result": "Success", "data": results_array[0]}
        db.close()
        return results_json
    else:
        return JSONResponse(status_code=500, content={"result": "Fail", "INFO" : "Error With stock_id {}".format(stock_id)})