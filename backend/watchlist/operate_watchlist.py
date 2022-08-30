from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import pymysql
from utils.StaticConfig import MYSQLStaticConfig
from utils.sql_fetch_json import sql_fetch_json

router = APIRouter(
    prefix = '/watchlist'
)

class watchlist(BaseModel):
    account_id : int
    stock_id : int

@router.post('/add/')
async def watchlist_add(data: watchlist):
    db = pymysql.connect(host=MYSQLStaticConfig.IP,port=MYSQLStaticConfig.PORT,user=MYSQLStaticConfig.USER,password=MYSQLStaticConfig.PASSWORD,database=MYSQLStaticConfig.DATABASE)
    cursor = db.cursor()
    sql = "INSERT INTO watchlist (account_id, stock_id) VALUES ({},{})".format(data.account_id, data.stock_id)
    try:
        cursor.execute(sql)
        db.commit()
    except pymysql.MySQLError as e:
        print("Error {} for execute sql: {}".format(e, sql))
        db.rollback()
        db.close()
        return JSONResponse(status_code=500, content={"result": "Fail", "INFO" : "Error With Database Server"})
    results_json = {"result": "Success"}
    db.close()
    return results_json

@router.post('/remove/')
async def watchlist_remove(data: watchlist):
    db = pymysql.connect(host=MYSQLStaticConfig.IP,port=MYSQLStaticConfig.PORT,user=MYSQLStaticConfig.USER,password=MYSQLStaticConfig.PASSWORD,database=MYSQLStaticConfig.DATABASE)
    cursor = db.cursor()
    sql = "DELETE FROM watchlist where account_id = {} AND stock_id = {}".format(data.account_id, data.stock_id)
    try:
        cursor.execute(sql)
        db.commit()
    except pymysql.MySQLError as e:
        print("Error {} for execute sql: {}".format(e, sql))
        db.rollback()
        db.close()
        return JSONResponse(status_code=500, content={"result": "Fail", "INFO" : "Error With Database Server"})
    results_json = {"result": "Success"}
    db.close()
    return results_json