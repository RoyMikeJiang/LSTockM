from fastapi import APIRouter
from fastapi.responses import JSONResponse
import pymysql
from utils.StaticConfig import MYSQLStaticConfig
from utils.sql_fetch_json import sql_fetch_json

router = APIRouter(
    prefix = '/price/history'
)

@router.get('/')
async def history_price(stock_id: int, start_date: str, end_date: str):
    db = pymysql.connect(host=MYSQLStaticConfig.IP,port=MYSQLStaticConfig.PORT,user=MYSQLStaticConfig.USER,password=MYSQLStaticConfig.PASSWORD,database=MYSQLStaticConfig.DATABASE)
    cursor = db.cursor()
    sql = "SELECT * FROM ohlcv WHERE stock_id={} AND price_date >= '{}' AND price_date <= '{}'".format(stock_id,start_date, end_date)
    results_array = []
    try:
        cursor.execute(sql)
        results_array = await sql_fetch_json(cursor)
    except pymysql.MySQLError as e:
        print("Error {} for execute sql: {}".format(e, sql))
        db.close()
        return JSONResponse(status_code=500, content={"result": "Fail", "INFO" : "Error With Database Server"})
    results_json = {"result": "Success", "data": results_array}
    db.close()
    return results_json