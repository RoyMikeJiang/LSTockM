from fastapi import APIRouter, Request
from pydantic import BaseModel
import pymysql
from utils.StaticConfig import MYSQLStaticConfig
from utils.sql_fetch_json import sql_fetch_json
from fastapi.responses import JSONResponse

async def varify_token(token):
    # print(token)
    # return [{'userid':1, 'username':'roy', 'level':3}]
    db = pymysql.connect(host=MYSQLStaticConfig.IP,port=MYSQLStaticConfig.PORT,user=MYSQLStaticConfig.USER,password=MYSQLStaticConfig.PASSWORD,database=MYSQLStaticConfig.DATABASE)
    cursor = db.cursor()
    sql = "SELECT account_id, username FROM account_info where token='{}' AND now()<=token_time".format(token)
    result_array = []
    try:
        cursor.execute(sql)
        result_array = await sql_fetch_json(cursor)
    except pymysql.MySQLError as e:
        print("Error {} for execute sql: {}".format(e, sql))
    db.close()
    return result_array

router = APIRouter(
    prefix = '/account/verify'
)

@router.get('/')
async def verify_token(request : Request):
    # print(request.cookies)
    user_info = await varify_token(request.cookies.get('token'))
    if len(user_info) == 0:
        return JSONResponse(status_code=401, content={"result" : "UnAuth"})
    else:
        return {"result" :"Success", "account_info": user_info[0]}
