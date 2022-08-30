from lib2to3.pgen2.token import OP
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import pymysql
from utils.StaticConfig import MYSQLStaticConfig
from utils.sql_fetch_json import sql_fetch_list
from typing import Optional

router = APIRouter(
    prefix = '/price/recent'
)

@router.get('/')
async def history_price(stock_id: int, period: Optional[int] = 30, date: Optional[bool] = False):
    db = pymysql.connect(host=MYSQLStaticConfig.IP,port=MYSQLStaticConfig.PORT,user=MYSQLStaticConfig.USER,password=MYSQLStaticConfig.PASSWORD,database=MYSQLStaticConfig.DATABASE)
    cursor = db.cursor()
    sql = "SELECT adj_close_price FROM ohlcv WHERE stock_id={} ORDER BY price_date DESC LIMIT {}".format(stock_id, period)
    results_array = []
    try:
        cursor.execute(sql)
        results_array = await sql_fetch_list(cursor)
    except pymysql.MySQLError as e:
        print("Error {} for execute sql: {}".format(e, sql))
        db.close()
        return JSONResponse(status_code=500, content={"result": "Fail", "INFO" : "Error With Database Server"})
    if(date):
        sql = "SELECT price_date FROM ohlcv WHERE stock_id={} ORDER BY price_date DESC LIMIT {}".format(stock_id, period)
        date_array = []
        try:
            cursor.execute(sql)
            date_array = await sql_fetch_list(cursor)
        except pymysql.MySQLError as e:
            print("Error {} for execute sql: {}".format(e, sql))
            db.close()
            return JSONResponse(status_code=500, content={"result": "Fail", "INFO" : "Error With Database Server"})
        results_json = {"result": "Success", "data": results_array[::-1], "date": date_array[::-1]}
        db.close()
        return results_json
    else:
        results_json = {"result": "Success", "data": results_array[::-1]}
        db.close()
        return results_json