from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import Optional
import pymysql
from utils.StaticConfig import MYSQLStaticConfig
from utils.sql_fetch_json import sql_fetch_json
import threading
import yfinance as yf
import sys
from pydantic import BaseModel
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import numpy as np
from torch.autograd import Variable
import os
from torchvision import transforms
import time

proxy = "http://127.0.0.1:7890"
# proxy = ""

## Database Config
# host = 'localhost'
# # host = 'localhost'
# user = 'root'
# password = 'password'
# database = 'lstockm'
host=MYSQLStaticConfig.IP
user=MYSQLStaticConfig.USER
password=MYSQLStaticConfig.PASSWORD
database=MYSQLStaticConfig.DATABASE


def changeStatus(account_id, stock_symbol, status, cursor, db):
    sql = "UPDATE predict_request SET status = '{}' WHERE account_id = {} AND stock_symbol = '{}'".format(status, account_id, stock_symbol)
    try:
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        print("Error when updating the status. sql: {}".format(sql))
        db.rollback()
    return
        
def thread_getData(account_id, stock_symbol):
    ticker = yf.Ticker(stock_symbol)
    ticker._get_fundamentals(proxy=proxy)
    # ticker._get_fundamentals(proxy=proxy)
    market_data = ticker.history(period="10y", auto_adjust=False, proxy=proxy)
    if(len(market_data)==0):
        print("Error with getting price. ticker_id: {}".format(stock_id))
        changeStatus(account_id, stock_symbol, "GetPriceError!", cursor, db)

    ## Database Connection Init
    db = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    cursor = db.cursor()

    if('symbol' not in ticker.info.keys()):
        print("No Stock Founded for Symbol: {}".format(stock_symbol))
        changeStatus(account_id, stock_symbol, "No Stock Founded", cursor, db)
        db.close()
        exit(1)

    sql = "SELECT exchange_id FROM exchange_info WHERE exchange_name = '{}'".format(ticker.info['exchange'])
    cursor.execute(sql)
    exchange_id = cursor.fetchall()
    if len(exchange_id)==0:
        sql = "INSERT INTO exchange_info (exchange_name, currency) VALUES ('{}','{}')".format(ticker.info['exchange'], ticker.info['currency'])
        try:
            cursor.execute(sql)
            exchange_id = cursor.lastrowid
            db.commit()
        except Exception as e:
            print("Error with inserting the exchange info. sql: {} ticker_symbol: {}".format(sql, ticker.info['symbol']))
            db.rollback()
            changeStatus(account_id, stock_symbol, "Server Error!", cursor, db)
            db.close()
            exit(1)
    else:
        exchange_id = exchange_id[0][0]
    
    sql = "INSERT INTO stock_info (stock_symbol, company_name, exchange_id) VALUES ('{}','{}',{})".format(ticker.info['symbol'], ticker.info['shortName'], exchange_id)
    stock_id = 0
    try:
        cursor.execute(sql)
        stock_id = cursor.lastrowid
        db.commit()
    except Exception as e:
        print("Error with inserting the stock info. sql: {} ticker_symbol: {}".format(sql, ticker.info['symbol']))
        db.rollback()
        changeStatus(account_id, stock_symbol, "Server Error!", cursor, db)
        db.close()
        exit(1)

    sql = "UPDATE predict_request SET stock_id = {} WHERE account_id = {} AND stock_symbol = '{}'".format(stock_id, account_id, stock_symbol)
    try:
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        print("Error when updating the status. sql: {}".format(sql))
        db.rollback()
        changeStatus(account_id, stock_symbol, "Server Error!", cursor, db)
        db.close()
        exit(1)

    sql = "INSERT INTO current_data (stock_id) VALUES ({})".format(stock_id)
    try:
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        print("Error when init the current_data. stock_id: {}".format(stock_id))
        db.rollback()
        changeStatus(account_id, stock_symbol, "Server Error!", cursor, db)
        db.close()
        exit(1)
    
    for i in range(5):
        sql = "INSERT INTO predict_price (stock_id, predict_price, predict_id) VALUES ({}, 1, {})".format(stock_id, i+1)
        try:
            cursor.execute(sql)
            db.commit()
        except Exception as e:
            print("Error when init the predict_price. stock_id: {}".format(stock_id))
            db.rollback()
            changeStatus(account_id, stock_symbol, "Server Error!", cursor, db)
            db.close()
            exit(1)
    
    sql = ''
    for index, row in market_data.iterrows():
        try:
            sql = "INSERT INTO ohlcv (stock_id, data_source_id, price_date, open_price, high_price, low_price, close_price, adj_close_price, volume) VALUES ({}, 1, '{}', {}, {}, {}, {}, {}, {})".format(stock_id, index, row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'], row['Volume'])
            cursor.execute(sql)
            db.commit()
        except Exception as e:
            db.rollback()
            changeStatus(account_id, stock_symbol, "Server Error!", cursor, db)
            print("Error when updating the stock(_id): {}\n SQL statement: {}\n Error info: {}".format(stock_id, sql, e))
            continue
    
    changeStatus(account_id, stock_symbol, "Data Loaded", cursor, db)
    db.close()
    thread_train(account_id, stock_id ,stock_symbol)

# region LSTM Basic Functions
class lstm(nn.Module):
    '''
    
    自定义LSTM模型 
    init function：LSTM+全连接层，共两层
    forward：模型前向传播函数，提取最后一次循环的最后一个维度的值作为输出
    
    '''
 
    def __init__(self,input_size=5,hidden_size=32,output_size=1):
        super(lstm, self).__init__()
        self.hidden_size=hidden_size
        self.input_size=input_size
        self.output_size=output_size
        self.rnn=nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,batch_first=True)
        self.linear=nn.Linear(self.hidden_size,self.output_size)
 
 
    def forward(self,x):
        out,(hidden,cell)=self.rnn(x)
        a, b, c = hidden.shape
        out=self.linear(hidden.reshape(a*b,c))
        return out
    

class my_dataset(Dataset):
    '''
    
    自定义dataset类，对样本进行封装和处理
    
    '''
    def __init__(self,ini_x,ini_y,transform=None):
        self.x=ini_x
        self.y=ini_y
        self.tranform = transform
 
    def __getitem__(self,index):
        x_out=self.x[index]
        y_out=self.y[index]
        if self.tranform !=None:
            return self.tranform(x_out),y_out
        return x_out,y_out
 
    def __len__(self):
        return len(self.x)    

    
### 辅助函数###
###############
def data_process(raw_data):
    '''
    数据处理：提取需要的数据字段，对数据的每个维度进行正则化，获取最小值和最大值
    raw_data：输入的原始stocks数据表格，以dataframe形式读取
    
    '''
    try:
        stocks=raw_data.sort_index(ascending=True)
        stocks=stocks[["Open","Close","High","Low"]]

        df_diff=stocks.diff().dropna()
        df_baseline=stocks[["Close"]]
        
    except TypeError:
        print("Type error, 'stocks' should be data_frame")
    
    return (df_diff,df_baseline)


def gain_dataset(data, divide=0.7):
    sequence=5
    stocks, base = data_process(data)
    total_len=stocks.shape[0]
    
    X=[]
    Y=[]
    baseline=[]
    for i in range(total_len-sequence):
        X.append(np.array(stocks.iloc[i:(i+sequence),1].values,dtype=np.float32).reshape(-1,1))
        Y.append(np.array(stocks.iloc[(i+sequence),1],dtype=np.float32))
        baseline.append(np.array(base.iloc[(i+sequence),0],dtype=np.float32))
    
    train_x,train_y = X[:int(divide*total_len)], Y[:int(divide*total_len)]
    test_x,test_y = X[int(divide*total_len):], Y[int(divide*total_len):]
    train_base, test_base = baseline[int(divide*total_len):], baseline[int(divide*total_len):]

    train_loader=DataLoader(dataset=my_dataset(train_x,train_y,transform=transforms.ToTensor()), batch_size=12, shuffle=True)
    test_loader=DataLoader(dataset=my_dataset(test_x,test_y), batch_size=10, shuffle=True)
    
    return train_loader, test_loader, train_base, test_base



### 以下为可调用的功能函数###
#############################
def train(data, pth_save, pth_xtrt='', max_epoch=500, lr=0.001):
    
    # 构建并训练模型
    
    train_loader, test_loader, train_base, test_base = gain_dataset(data)
    
    if pth_xtrt=='':       
        model=lstm(input_size=1,hidden_size=32,output_size=1) 
    else:
        model=lstm(input_size=1,hidden_size=32,output_size=1) 
        model.load_state_dict(torch.load(pth_xtrt))    
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=lr)
 
    loss_ls=[]
    for i in range(max_epoch):
        total_loss=0
        for idx,(data,label) in enumerate(train_loader):
            data=data.squeeze(1)
            pred=model(Variable(data))
            label=label.unsqueeze(1)
            loss=criterion(pred,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        # print(i+1,"Loss: ",total_loss)
        loss_ls.append(total_loss)

    torch.save(model.state_dict(), pth_save)
    
    return loss_ls

def predict(data, pth):
    '''
    
    调用训练好的LSTM模型，通过获取最近5日的股票数据输入模型中，预测第6日的收盘价格     
    stocks：以data_frame形式读取的股票数据
    pth：对应股票的LSTM模型路径
    
    '''
    sequence = 5    
    stocks, base = data_process(data)
    base_ls=base.values.tolist()[-1]    
    X=np.array(stocks.iloc[:sequence,1].values,dtype=np.float32)

    model=lstm(input_size=1,hidden_size=32,output_size=1) 
    model.load(pth)
    pred_ls=[]
    for i in range(5):
        input_data=torch.tensor(X[-5:]).view(1,-1,1)
        live_pred=model(input_data)
        X = np.append(X, np.array(live_pred.data.squeeze(1)))
        pred = live_pred.data.squeeze(1).tolist()
        pred_ls.extend(pred)
        base_ls.append(pred[0]+base_ls[-1])

    predict = np.array(pred_ls, dtype=np.float32)+np.array(base_ls[:5], dtype=np.float32)
    
    return predict

# endregion

def getNextFiveDays(day = datetime.now()):
    five_date_list = []
    for i in range(5):
        five_date_list.append(getNextWeekDay(day+timedelta(days=i)).strftime("%Y-%m-%d"))
    return five_date_list

def getNextWeekDay(day=datetime.now()):
    if day.weekday()>=4:
      dayStep=7-day.weekday()
    else:
      dayStep=1
    nextWorkDay = day + timedelta(days=dayStep)
    return nextWorkDay

def thread_train(account_id, stock_id, stock_symbol, data_length = 900):

    ## Database Connection Init
    db = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    cursor = db.cursor()
    price_data = []
    sql = "SELECT open_price, close_price, high_price, low_price, volume FROM ohlcv WHERE stock_id = {} ORDER BY price_date DESC LIMIT {};".format(stock_id, data_length)
    try:
        cursor.execute(sql)
        price_data = cursor.fetchall()
    except Exception as e:
        print("Error when getting data from sql : {} stock_id : {}".format(sql, stock_id))
        changeStatus(account_id, stock_symbol, "Server Error!", cursor, db)
        db.close()
        exit(1)
    df = pd.DataFrame(price_data, columns = ["Open","Close","High","Low","Volume"])
    train(data=df, pth_save='../AutomaticUpdateData/model/{}.pth'.format(stock_id))
    
    sql = "INSERT INTO lstm_path (stock_id, model_path) VALUES ({}, 'model/{}.pth')".format(stock_id, stock_id)
    try:
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        print("Error when insert path into lstm_path. sql: {} stock_id: {}".format(sql, stock_id))
        db.rollback()
        changeStatus(account_id, stock_symbol, "Server Error!", cursor, db)
        db.commit()
        exit(1)
    changeStatus(account_id, stock_symbol, "LSTM Trained", cursor, db)
    db.close()
    thread_predict(account_id, stock_id, stock_symbol)

def thread_predict(account_id, stock_id, stock_symbol):
    ## Database Connection Init
    db = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    cursor = db.cursor()

    ## Get the Stocks List
    stock_list = ''
    try:
        cursor.execute("SELECT stock_id, stock_symbol, model_path FROM stock_info si LEFT OUTER JOIN lstm_path USING (stock_id) WHERE stock_id={}".format(stock_id))
        stock_list = cursor.fetchall()
    except Exception as e:
        print(e)
        changeStatus(account_id, stock_symbol, "Server Error!", cursor, db)
        db.close()
        exit(1)

    for stock_info in stock_list:
        if(stock_info[2]==None):
            continue
        # print(stock_info)
        cursor.execute("SELECT open_price, close_price, high_price, low_price, volume FROM ohlcv WHERE stock_id = {} ORDER BY price_date DESC LIMIT 5;".format(stock_info[0]))
        price_data = cursor.fetchall()
        df = pd.DataFrame(price_data, columns = ["Open","Close","High","Low","Volume"])
        # print(df)
        predict_price = predict(df, "../AutomaticUpdateData/"+stock_info[2])
        predict_date = getNextFiveDays()
        for i in range(5):
            sql = "UPDATE predict_price SET predict_time = now(), price_date = '{}' , predict_price = {} WHERE stock_id={} AND predict_id={}".format(predict_date[i], predict_price[i], stock_info[0], i+1)
            # print(sql)
            try:
                cursor.execute(sql)
                db.commit()
            except Exception as e:
                print("Error when updating the stock(_id): {}\n SQL statement: {}\n Error info: {}".format(stock_info[0], sql, e))
                changeStatus(account_id, stock_symbol, "Server Error!", cursor, db)
                db.rollback()
    sql = "UPDATE predict_request SET finish_time = now(), status = 'Predict Finished' WHERE account_id = {} AND stock_symbol = '{}'".format(account_id, stock_symbol)
    try:
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        print("Error when updating the status. sql: {}".format(sql))
        db.rollback()
        changeStatus(account_id, stock_symbol, "Server Error!", cursor, db)
        db.close()
        exit(1)
    db.close()

router = APIRouter(
    prefix = '/predict/request'
)

class predictRequest(BaseModel):
    account_id : int
    stock_symbol : str

class myThread (threading.Thread):
    def __init__(self, threadID, name, account_id, stock_symbol):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.account_id = account_id
        self.stock_symbol = stock_symbol
    def run(self):
        thread_getData(self.account_id, self.stock_symbol)
    def __exit__(self, t, v, tb):
        thread_list[self.threadID] = 'Empty'
        self.release()

thread_list = ['Empty', 'Empty', 'Empty']

def get_thread_id(threadName):
    for i in range(len(thread_list)):
        if thread_list[i] == 'Empty':
            thread_list[i] = threadName
            return i
    return -1


@router.post('/')
async def request_predict_data(data: predictRequest):
    db = pymysql.connect(host=MYSQLStaticConfig.IP,port=MYSQLStaticConfig.PORT,user=MYSQLStaticConfig.USER,password=MYSQLStaticConfig.PASSWORD,database=MYSQLStaticConfig.DATABASE)
    cursor = db.cursor()
    sql = "INSERT INTO predict_request (account_id, stock_symbol, status) VALUES ({},'{}','Started')".format(data.account_id, data.stock_symbol)
    try:
        cursor.execute(sql)
        db.commit()
    except pymysql.MySQLError as e:
        print("Error {} for execute sql: {}".format(e, sql))
        db.rollback()
        db.close()
        return JSONResponse(status_code=500, content={"result": "Fail", "INFO" : "Error With Database Server"})
    thread_id = get_thread_id(data.stock_symbol)
    thread = myThread(thread_id, data.stock_symbol, data.account_id, data.stock_symbol)
    thread.start()
    results_json = {"result": "Success"}
    db.close()
    return results_json