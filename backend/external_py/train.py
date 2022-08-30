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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--stock_id')
parser.add_argument('--pretrain_path', default='')
parser.add_argument('--data_length', default=900)

if __name__ == '__main__':
    import pymysql
    import sys

    args = parser.parse_args()

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
    price_data = []
    sql = "SELECT open_price, close_price, high_price, low_price, volume FROM ohlcv WHERE stock_id = {} ORDER BY price_date DESC LIMIT {};".format(args.stock_id, args.data_length)
    try:
        cursor.execute(sql)
        price_data = cursor.fetchall()
    except Exception as e:
        print("Error when getting data from sql : {} stock_id : {}".format(sql, args.stock_id))
        db.close()
        sys.exit(1)
    df = pd.DataFrame(price_data, columns = ["Open","Close","High","Low","Volume"])
    train(data=df, pth_save='../../AutomaticUpdateData/model/{}.pth'.format(args.stock_id), pth_xtrt=args.pretrain_path)
    
    if(args.pretrain_path == ''):
        sql = "INSERT INTO lstm_path (stock_id, model_path) VALUES ({}, 'model/{}.pth')".format(args.stock_id, args.stock_id)
        try:
            cursor.execute(sql)
            db.commit()
        except Exception as e:
            print("Error when insert path into lstm_path. sql: {} stock_id: {}".format(sql, args.stock_id))
            db.rollback()
            db.commit()
            sys.exit(1)
    db.close()



