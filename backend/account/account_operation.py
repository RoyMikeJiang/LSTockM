from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import Optional
import pymysql
from utils.StaticConfig import MYSQLStaticConfig
from utils.sql_fetch_json import sql_fetch_json
import yfinance as yf
from datetime import datetime
from pydantic import BaseModel

router = APIRouter(
    prefix = '/account'
)

class signUpInfo(BaseModel):
    username: str
    password: str

@router.post('/signup/')
async def signUp(data: signUpInfo):
    db = pymysql.connect(host=MYSQLStaticConfig.IP,port=MYSQLStaticConfig.PORT,user=MYSQLStaticConfig.USER,password=MYSQLStaticConfig.PASSWORD,database=MYSQLStaticConfig.DATABASE)
    cursor = db.cursor()
    sql = "INSERT INTO account_info (username, password) VALUES ('{}','{}')".format(data.username, data.password)
    try:
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        print("Error when create account. sql: {}".format(sql))
        db.rollback()
        db.close()
        return JSONResponse(status_code=500, content={"result": "Fail", "INFO" : "Error With Database Server"})
    db.close()
    return {"result": "Success"}

@router.get('/info/')
async def account_info(account_id: int):
    db = pymysql.connect(host=MYSQLStaticConfig.IP,port=MYSQLStaticConfig.PORT,user=MYSQLStaticConfig.USER,password=MYSQLStaticConfig.PASSWORD,database=MYSQLStaticConfig.DATABASE)
    cursor = db.cursor()
    sql = "SELECT username FROM account_info WHERE account_id = {}".format(account_id)
    result = []
    try:
        cursor.execute(sql)
        result = await sql_fetch_json(cursor)
    except Exception as e:
        print("Error when create account.")
        db.close()
        return JSONResponse(status_code=500, content={"result": "Fail", "INFO" : "Error With Database Server"})
    db.close()
    return {"result": "Success", "username" : result[0]['username']}