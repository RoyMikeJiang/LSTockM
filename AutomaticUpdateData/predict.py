from turtle import back
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import numpy as np
from torch.autograd import Variable
import os, sys
from torchvision import transforms

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
def train(data, pth_save, pth_xtrt='', max_epoch=100, lr=0.001):
    
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
        print(i+1,"Loss: ",total_loss)
        loss_ls.append(total_loss)

    torch.save(model.state_dict(), pth_save)
    
    return loss_ls
    
    
def test(data, pth):
    
    # 测试模型
    
    stocks, base = data_process(data)
    train_loader, test_loader, train_base, test_base = gain_dataset(data)
    model=torch.load(pth)
    
    test_preds=[]
    test_labels=[]
    for idx, (x, label) in enumerate(test_loader):
        pred=model(x)
        test_preds.extend(pred.data.squeeze(1).tolist())
        test_labels.extend(label.tolist())
    
    preds=pd.DataFrame(test_preds,columns=["Close"])
    labels=pd.DataFrame(test_labels,columns=["Close"])
    base_line=pd.DataFrame(test_base,columns=["Close"])
    train_shift = base_line["Close"].shift(1)
#     print(base_line.head())
    predi = preds["Close"].add(train_shift).dropna()
#     print(predi.head())
    predict = predi.values.tolist()
    
    
    return predict, test_base
    

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
    model.load_state_dict(torch.load(pth))
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

if __name__ == '__main__':
    import pymysql
    import sys

    ## Database Config
    host = 'localhost'
    # host = 'localhost'
    user = 'root'
    password = 'password'
    database = 'lstockm'

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
        cursor.execute("SELECT stock_id, stock_symbol, model_path FROM stock_info si LEFT OUTER JOIN lstm_path USING (stock_id);")
        stock_list = cursor.fetchall()
    except Exception as e:
        print(e)
        db.close()
        sys.exit(1)

    for stock_info in stock_list:
        print(stock_info)
        if(stock_info[2]==None):
            continue
        # print(stock_info)
        cursor.execute("SELECT open_price, close_price, high_price, low_price, volume FROM ohlcv WHERE stock_id = {} ORDER BY price_date DESC LIMIT 5;".format(stock_info[0]))
        price_data = cursor.fetchall()
        df = pd.DataFrame(price_data, columns = ["Open","Close","High","Low","Volume"])
        cursor.execute("SELECT price_date FROM ohlcv where stock_id = {} ORDER BY price_date DESC LIMIT 1;".format(stock_info[0]))
        last_date = cursor.fetchall()[0][0]
        # print(df)
        predict_price = predict(df, stock_info[2])
        predict_date = getNextFiveDays(last_date)
        for i in range(5):
            sql = "UPDATE predict_price SET predict_time = now(), price_date = '{}' , predict_price = {} WHERE stock_id={} AND predict_id={}".format(predict_date[i], predict_price[i], stock_info[0], i+1)
            # print(sql)
            try:
                cursor.execute(sql)
                db.commit()
            except Exception as e:
                print("Error when updating the stock(_id): {}\n SQL statement: {}\n Error info: {}".format(stock_info[0], sql, e))
                db.rollback()
    db.close()