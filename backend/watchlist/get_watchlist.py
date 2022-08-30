from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import Optional
import pymysql
from utils.StaticConfig import MYSQLStaticConfig
from utils.sql_fetch_json import sql_fetch_json

router = APIRouter(
    prefix = '/watchlist/get'
)

@router.get('/')
async def watchlist_get(account_id: int):
    db = pymysql.connect(host=MYSQLStaticConfig.IP,port=MYSQLStaticConfig.PORT,user=MYSQLStaticConfig.USER,password=MYSQLStaticConfig.PASSWORD,database=MYSQLStaticConfig.DATABASE)
    cursor = db.cursor()
    sql = "SELECT stock_id, stock_symbol, company_name, exchange_name, currency FROM watchlist w LEFT OUTER JOIN stock_info si USING (stock_id) LEFT OUTER JOIN exchange_info ei USING(exchange_id) WHERE account_id = {} ".format(account_id)
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