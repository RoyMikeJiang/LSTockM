from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import Optional
import pymysql
from utils.StaticConfig import MYSQLStaticConfig
from utils.sql_fetch_json import sql_fetch_json

router = APIRouter(
    prefix = '/stock/info'
)

@router.get('/')
async def stock_info(stock_id: Optional[int] = 0, stock_symbol: Optional[str] = None):
    '''
        If received both two parameters, sql will only use the stock_id which is more specific.
    '''
    if stock_id == 0 and stock_symbol == None :
        return JSONResponse(status_code=500, content={"result": "Fail", "INFO" : "Please at least use one query parameters!"})
    db = pymysql.connect(host=MYSQLStaticConfig.IP,port=MYSQLStaticConfig.PORT,user=MYSQLStaticConfig.USER,password=MYSQLStaticConfig.PASSWORD,database=MYSQLStaticConfig.DATABASE)
    cursor = db.cursor()
    sql = "SELECT stock_id, stock_symbol, company_name, exchange_name, currency FROM stock_info si LEFT OUTER JOIN exchange_info ei USING(exchange_id) WHERE "
    if stock_id != 0:
        sql += "stock_id={}".format(stock_id)
    elif stock_symbol != None:
        sql += "stock_symbol LIKE '%{}%'".format(stock_symbol)
    else:
        pass
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