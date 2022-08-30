from fastapi import APIRouter
from fastapi.responses import JSONResponse
import pymysql
from utils.StaticConfig import MYSQLStaticConfig
from utils.sql_fetch_json import sql_fetch_json

router = APIRouter(
    prefix = '/watchlist'
)

@router.get('/check/')
async def watchlist_check(account_id, stock_id):
    db = pymysql.connect(host=MYSQLStaticConfig.IP,port=MYSQLStaticConfig.PORT,user=MYSQLStaticConfig.USER,password=MYSQLStaticConfig.PASSWORD,database=MYSQLStaticConfig.DATABASE)
    cursor = db.cursor()
    sql = "SELECT COUNT(*) as LEN FROM watchlist WHERE account_id = {} AND stock_id = {}".format(account_id, stock_id)
    result_array = []
    try:
        cursor.execute(sql)
        result_array = await sql_fetch_json(cursor)
    except pymysql.MySQLError as e:
        print("Error {} for execute sql: {}".format(e, sql))
        db.close()
        return JSONResponse(status_code=500, content={"result": "Fail", "INFO" : "Error With Database Server"})
    results_json = {"result": "Success", "check" : True if result_array[0]['LEN']==1 else False}
    db.close()
    return results_json