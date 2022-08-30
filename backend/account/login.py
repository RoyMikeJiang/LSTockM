from fastapi import APIRouter
from pydantic import BaseModel
import pymysql
from utils.StaticConfig import MYSQLStaticConfig
from utils.sql_fetch_json import sql_fetch_json
import hashlib
import os

router = APIRouter(
    prefix = '/account/login'
)

class LoginInfo(BaseModel):
    username: str
    password: str

def generate_token():
    return hashlib.sha1(os.urandom(24)).hexdigest()

@router.post('/')
async def login(data: LoginInfo):
    db = pymysql.connect(host=MYSQLStaticConfig.IP,port=MYSQLStaticConfig.PORT,user=MYSQLStaticConfig.USER,password=MYSQLStaticConfig.PASSWORD,database=MYSQLStaticConfig.DATABASE)
    cursor = db.cursor()
    sql = "SELECT account_id, username FROM account_info WHERE username = '{}' AND password = '{}'".format(data.username, data.password)
    results_array = []
    token = ''
    try:
        cursor.execute(sql)
        results_array = await sql_fetch_json(cursor)
        if(len(results_array)==0):
            db.close()
            return {"result" : "ErrorPassword"}
        token = generate_token()
        sql1 = "UPDATE account_info SET token='{}', token_time = DATE_ADD(NOW(), INTERVAL 1 HOUR) WHERE username='{}'".format(token, data.username)
        try:
            cursor.execute(sql1)
        except pymysql.MySQLError as e:
            print("Error {} for execute sql: {}".format(e, sql1))
            db.rollback()
            db.close()
            return {"result": "Fail"}
    except pymysql.MySQLError as e:
        print("Error {} for execute sql: {}".format(e, sql))
        db.close()
        return {"result" : "Fail"}
    results_json = {"result":"Success", "data":results_array[0], 'token': token}
    db.commit()
    db.close()
    return results_json