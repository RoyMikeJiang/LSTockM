from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

from price import history_price, recent_price, current_price
from stock import stock_info, random_stock
from statistics import current_statistics, get_var
from predict import get_predict, request_predict, get_request
from watchlist import get_watchlist, operate_watchlist, check_watchlist
from account import cookie, account_operation, login
from suggestion import get_suggestion

app.include_router(history_price.router)
app.include_router(recent_price.router)
app.include_router(current_price.router)

app.include_router(stock_info.router)
app.include_router(random_stock.router)

app.include_router(current_statistics.router)
app.include_router(get_var.router)

app.include_router(get_predict.router)
app.include_router(request_predict.router)
app.include_router(get_request.router)

app.include_router(get_watchlist.router)
app.include_router(operate_watchlist.router)
app.include_router(check_watchlist.router)

app.include_router(cookie.router)
app.include_router(account_operation.router)
app.include_router(login.router)

app.include_router(get_suggestion.router)

import pymysql
from utils.StaticConfig import MYSQLStaticConfig
from utils.sql_fetch_json import sql_fetch_json
async def varify_token(token):
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

Level_List = [
    ['/account/info/']
    ]

@app.middleware("http")
async def test(request : Request, call_next):
    expect_check = False
    # print(request.url.path)
    for i in range(len(Level_List)):
        if request.url.path in Level_List[i]:
            expected_level = True
            break
    if(expect_check):
        user_info = await varify_token(request.cookies.get('token'))
        if len(user_info) == 0:
            return JSONResponse(status_code=401, content={"result" : "UnAuth"})
    response = await call_next(request)
    return response

@app.get("/")
async def root():
    return {"message": "LSTockM Backend"}