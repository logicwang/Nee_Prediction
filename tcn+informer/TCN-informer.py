import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm

import torch
import sys
print(torch.__version__)
sys.stdout.flush()
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

plt.rc('font', family='Arial')
plt.style.use("ggplot")
# 自己写的函数文件functionfile.py
# 如果需要调整TSlib-test.ipynb文件的路径位置 注意同时调整导入的路径
from models import TCNInformer
from utils.timefeatures import time_features

# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def tslib_data_loader(window, length_size, batch_size, data, data_mark):
    """
    数据加载器函数，用于加载和预处理时间序列数据，以用于训练模型。

    仅仅适用于 多变量预测多变量（可以单独取单变量的输出），或者单变量预测单变量
    也就是y里也会有外生变量？？

    参数:
    - window: 窗口大小，用于截取输入序列的长度。
    - length_size: 目标序列的长度。
    - batch_size: 批量大小，决定每个训练批次包含的数据样本数量。
    - data: 输入时间序列数据。
    - data_mark: 输入时间序列的数据标记，用于辅助模型训练或增加模型的多样性。

    返回值:
    - dataloader: 数据加载器，用于批量加载处理后的训练数据。
    - x_temp: 处理后的输入数据。
    - y_temp: 处理后的目标数据。
    - x_temp_mark: 处理后的输入数据的标记。
    - y_temp_mark: 处理后的目标数据的标记。
    """

    # 构建模型的输入
    seq_len = window
    sequence_length = seq_len + length_size
    num_samples = len(data) - sequence_length + 1
    print(f"TSLIB data generator start, seq_len: {seq_len}, length_size: {length_size}, total samples: {num_samples}")
    
    # 使用预分配空间代替列表推导式，极大提升大切片场景下的性能
    result = np.empty((num_samples, sequence_length, data.shape[1]), dtype=np.float32)
    result_mark = np.empty((num_samples, sequence_length, data_mark.shape[1]), dtype=np.float32)
    
    for i in range(num_samples):
        result[i] = data[i : i + sequence_length]
        result_mark[i] = data_mark[i : i + sequence_length]
        
    print("TSLIB window splicing finished...")

    # 划分x与y
    x_temp = result[:, :-length_size]
    y_temp = result[:, -(length_size + int(window / 2)):]

    x_temp_mark = result_mark[:, :-length_size]
    y_temp_mark = result_mark[:, -(length_size + int(window / 2)):]

    # 转换为Tensor和数据类型
    x_temp = torch.tensor(x_temp).type(torch.float32)
    x_temp_mark = torch.tensor(x_temp_mark).type(torch.float32)
    y_temp = torch.tensor(y_temp).type(torch.float32)
    y_temp_mark = torch.tensor(y_temp_mark).type(torch.float32)

    ds = TensorDataset(x_temp, y_temp, x_temp_mark, y_temp_mark)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return dataloader, x_temp, y_temp, x_temp_mark, y_temp_mark


def model_train(net, train_loader, length_size, optimizer, criterion, num_epochs, device, print_train=False):
    """
    训练模型并应用早停机制。

    参数:
        net (torch.nn.Module): 待训练的模型。
        train_loader (torch.utils.data.DataLoader): 训练数据加载器。
        length_size (int): 输出序列的长度。
        optimizer (torch.optim.Optimizer): 优化器。
        criterion (torch.nn.Module): 损失函数。
        num_epochs (int): 总训练轮数。
        device (torch.device): 设备（CPU或GPU）。
        print_train (bool, optional): 是否在训练中打印进度，默认为False。
    返回:
        net (torch.nn.Module): 训练好的模型。
        train_loss (list): 训练过程中每个epoch的平均训练损失列表。
        best_epoch (int): 达到最佳验证损失的epoch数。
    """

    train_loss = []  # 用于记录每个epoch的平均训练损失
    print_frequency = 1 #num_epochs / 20  # 计算打印训练状态的频率

    for epoch in range(num_epochs):
        total_train_loss = 0  # 初始化一个epoch的总损失

        net.train()  # 将模型设置为训练模式
        loop = tqdm(train_loader, total=len(train_loader), leave=True, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for i, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(loop):
            datapoints, labels, datapoints_mark, labels_mark = datapoints.to(device), labels.to(
                device), datapoints_mark.to(device), labels_mark.to(device)
            optimizer.zero_grad()  # 清空梯度
            preds = net(datapoints, datapoints_mark, labels, labels_mark, None)  # 前向传播
            preds = preds[:, -length_size:, -1:] # 仅取目标变量通道
            labels = labels[:, -length_size:, -1:] # 仅取目标变量通道
            loss = criterion(preds, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            total_train_loss += loss.item()  # 累加损失值
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)  # 计算该epoch的平均损失
        train_loss.append(avg_train_loss)  # 将平均损失添加到列表中

        # 如果设置为打印训练状态，则按频率打印
        if print_train:
            if (epoch + 1) % print_frequency == 0:
                print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

    return net, train_loss, epoch + 1


def model_train_val(net, train_loader, val_loader, length_size, optimizer, criterion, scheduler, num_epochs, device,
                    early_patience=0.15, print_train=False):
    """
    训练模型并应用早停机制。

    参数:
        model (torch.nn.Module): 待训练的模型。
        train_loader (torch.utils.data.DataLoader): 训练数据加载器。
        val_loader (torch.utils.data.DataLoader): 验证数据加载器。
        optimizer (torch.optim.Optimizer): 优化器。
        criterion (torch.nn.Module): 损失函数。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器。
        num_epochs (int): 总训练轮数。
        device (torch.device): 设备（CPU或GPU）。
        early_patience (float, optional): 早停耐心值，默认为0.15 * num_epochs。
        print_train: 是否打印训练信息。
    返回:
        torch.nn.Module: 训练好的模型。
        list: 训练过程中每个epoch的平均训练损失列表。
        list: 训练过程中每个epoch的平均验证损失列表。
        int: 早停触发时的epoch数。
    """

    train_loss = []  # 用于记录每个epoch的平均损失
    val_loss = []  # 用于记录验证集上的损失，用于早停判断
    print_frequency = 1 #num_epochs / 20  # 计算打印频率

    early_patience_epochs = int(early_patience * num_epochs)  # 早停耐心值（转换为epoch数）
    best_val_loss = float('inf')  # 初始化最佳验证集损失
    early_stop_counter = 0  # 早停计数器

    for epoch in range(num_epochs):
        total_train_loss = 0  # 初始化一个epoch的总损失

        net.train()  # 将模型设置为训练模式
        loop = tqdm(train_loader, total=len(train_loader), leave=True, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for i, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(loop):
            datapoints, labels, datapoints_mark, labels_mark = datapoints.to(
                device), labels.to(device), datapoints_mark.to(device), labels_mark.to(device)
            optimizer.zero_grad()  # 清空梯度
            preds = net(datapoints, datapoints_mark, labels, labels_mark, None)
            preds = preds[:, -length_size:, -1:] # 仅取目标变量通道
            labels = labels[:, -length_size:, -1:] # 仅取目标变量通道
            loss = criterion(preds, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            total_train_loss += loss.item()  # 累加损失值
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)  # 计算本epoch的平均损失
        train_loss.append(avg_train_loss)  # 记录平均损失

        with torch.no_grad():  # 关闭自动求导以节省内存和提高效率
            total_val_loss = 0
            for val_x, val_y, val_x_mark, val_y_mark in val_loader:
                val_x, val_y, val_x_mark, val_y_mark = val_x.to(device), val_y.to(device), val_x_mark.to(
                    device), val_y_mark.to(device)  # 将数据移到GPU
                pred_val_y = net(val_x, val_x_mark, val_y, val_y_mark, None).squeeze()  # 前向传播
                val_y = val_y[:, -length_size:].squeeze()  # 注意这一步
                val_loss_batch = criterion(pred_val_y, val_y)  # 计算损失
                total_val_loss += val_loss_batch.item()

            avg_val_loss = total_val_loss / len(val_loader)  # 计算本epoch的平均验证损失
            val_loss.append(avg_val_loss)  # 记录平均验证损失

            scheduler.step(avg_val_loss)  # 更新学习率（基于当前验证损失）

        # 打印训练信息
        if print_train == True:
            if (epoch + 1) % print_frequency == 0:
                print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # 早停判断
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0  # 重置早停计数器
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_patience_epochs:
                print(f'Early stopping triggered at epoch {epoch + 1}.')
                break  # 早停

    net.train()  # 恢复训练模式
    return net, train_loss, val_loss, epoch + 1


# 计算点预测的评估指标
def cal_eval(y_real, y_pred):
    """
    输入参数:
    y_real - numpy数组，表示测试集的真实目标值。
    y_pred - numpy数组，表示预测的结果。

    输出:
    df_eval - pandas DataFrame对象
    """

    y_real, y_pred = np.array(y_real).ravel(), np.array(y_pred).ravel()

    #mse和rmse的代码更改过，注意分辨
    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred) #, squared=True)
    #rmse = mean_squared_error(y_real, y_pred) #, squared=False)  # RMSE and MAE are various on different scales
    #在scikit-learn的1.6.1版本中，计算均方根误差要用root_mean_squared_error函数
    #即rmse = root_mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred) * 100  # Note that dataset cannot have any 0 value.

    df_eval = pd.DataFrame({'R2': r2,
                            'MSE': mse, 'RMSE': rmse,
                            'MAE': mae, 'MAPE': mape},
                           index=['Eval'])

    return df_eval


def data_cleansing(df):
    """集成自 data_preprocessing.py 的清洗逻辑，兼容多个数据集"""
    # 确保日期格式正确并排序
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').reset_index(drop=True)
    
    # 记录原始列，排除日期列进行插值
    cols = [c for c in df.columns if c != 'date']
    
    # 1. 线性插值处理短时间缺失 (Limit = 6, 即 3 小时)
    df[cols] = df[cols].interpolate(method='linear', limit=6)
    
    # 2. 基于时间的插值处理中等时间缺失 (Limit = 48, 即 24 小时)
    if 'date' in df.columns:
        df.set_index('date', inplace=True)
        df[cols] = df[cols].interpolate(method='time', limit=48)
        df.reset_index(inplace=True)
    
    # 3. 前后填充处理剩余的长期缺失
    df[cols] = df[cols].ffill().bfill()
    
    return df


# 读取数据 - 切换到长江三角洲 DT 农田数据集（30分钟级）
data_path = 'data/Yangtze River Delta of China/DT_NEE(20141201-20171130).csv'
#data_path = 'data/Yangtze River Delta of China/SX_NEE(20150715-20190424).csv'
dataset_name = os.path.splitext(os.path.basename(data_path))[0]

print(f"开始读取数据集: {data_path} ...")
df_raw = pd.read_csv(data_path)
print(f"数据读取完成, 原始形状: {df_raw.shape}")

# 执行统一清洗
df = data_cleansing(df_raw)
print(f"数据清洗完成, 清洗后形状: {df.shape}")

# 统一目标列名，适应原代码逻辑
if 'Target' in df.columns:
    df.rename(columns={'Target': 'target'}, inplace=True)

# --- 方向 A 优化：特征工程 (Feature Engineering) ---
# 1. 滞后项 (Lagged Features): 捕捉生态滞后响应 (根据新数据集字段 K↓(辐射), Tair(温度), VPD(湿度差))
for col in ['K↓', 'Tair', 'VPD']:
    for lag in range(1, 4):
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

# 2. 差分项 (Differential Features): 捕捉环境突变速度
for col in ['K↓', 'Tair']:
    df[f'{col}_diff'] = df[col].diff()
print("特征工程完成(滞后+差分)...")

# 3. 清理 NaN (由于 shift 和 diff 产生)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True) # 必须重置索引以保证后续矩阵拼接的对齐

# 提取特征和目标变量
feature_cols = [c for c in df.columns if c not in ['date', 'target']]
data_target = df[['target']].values  # shape: (N, 1)
features = df[feature_cols].values   # shape: (N, num_features)

print("特征和目标变量提取完成，准备时间特征编码...")
# 时间戳特征提取 (30分钟数据，频率必须是 'h' 或 't' 以捕捉昼夜节律)
df_stamp = df[['date']].copy()
df_stamp['date'] = pd.to_datetime(df_stamp['date'])
data_stamp = time_features(df_stamp, timeenc=1, freq='h')
print("时间特征编码完成...")

# ==========================================
# --- 核心修复：防止数据泄露 (Data Leakage) ---
# **绝对要求：必须在调用任何 .fit() 之前切分训练集与测试集**
# ==========================================
data_length = len(df)
train_set = 0.8  # 保留 80% 训练集, 20% 测试集
train_size = int(train_set * data_length)

features_train = features[:train_size, :]
features_test = features[train_size:, :]
target_train = data_target[:train_size, :]
target_test = data_target[train_size:, :]
data_stamp_train = data_stamp[:train_size, :]
data_stamp_test = data_stamp[train_size:, :]
print("训练/测试集切分完成...")

# --- 方向 B 优化 (隔离模式)：PCA 降维去噪 ---
# 1. StandardScaler：【只对训练集 fitting，不对测试集学习参数】
print("开始进行数据标准化与 PCA 降维...")
scaler_pca = StandardScaler()
features_train_scaled = scaler_pca.fit_transform(features_train)
features_test_scaled = scaler_pca.transform(features_test) 

# 2. PCA降维：【同样，PCA 主成分网络绝不泄露给测试集】
pca = PCA(n_components=0.95) 
features_train_pca = pca.fit_transform(features_train_scaled)
features_test_pca = pca.transform(features_test_scaled)    
print(f"PCA 降维完成：特征维度从 {features_train.shape[1]} 降至 {features_train_pca.shape[1]}")

# 3. 将 PCA 过滤后的特征和原本的目标变量重新拼接成网络期待的输入矩阵
print("进行矩阵拼接与 MinMaxScaler 归一化...")
data_train_raw = np.concatenate((features_train_pca, target_train), axis=1)
data_test_raw = np.concatenate((features_test_pca, target_test), axis=1)

# --- 最终层隔离优化 ---
# 4. MinMaxScaler 归一化网络整体输入：【全局绝对极值必须局限于训练集】
scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train_raw)
data_test = scaler.transform(data_test_raw)           

data_train_mark = data_stamp_train
data_test_mark = data_stamp_test
data_dim = data_train.shape[1]  # 特征维度 (PCA维数) + 1 个目标维度

n_feature = data_dim
window = 96  # 输入序列: 过去 2 天的半小时数据 (48 * 2)
length_size = 48  # 预测未来的序列长度: 未来 1 天 (48 * 1)
batch_size = 128

print("准备封装 PyTorch DataLoader...")
train_loader, x_train, y_train, x_train_mark, y_train_mark = tslib_data_loader(window, length_size, batch_size,
                                                                               data_train, data_train_mark)
print("训练集 DataLoader 封装完成...")
test_loader, x_test, y_test, x_test_mark, y_test_mark = tslib_data_loader(window, length_size, batch_size, data_test,
                                                                          data_test_mark)
print("测试集 DataLoader 封装完成...")
"""
# 有验证集

# 数据归一化
scaler = MinMaxScaler()
data_inverse = scaler.fit_transform(np.array(data))
data_length = len(data_inverse)
train_ratio = 0.6
val_ratio = 0.8
# 6：2：2
window = 30  # 模型输入序列长度 过去30天的数据
length_size = 1  # 预测结果的序列长度  预测未来1天
train_size = int(data_length * train_ratio)
val_size = int(data_length * val_ratio)
data_train = data_inverse[:train_size, :]
data_train_mark = data_stamp[:train_size, :]
data_val = data_inverse[train_size: val_size, :]
data_val_mark = data_stamp[train_size: val_size, :]
data_test = data_inverse[val_size:, :]
data_test_mark = data_stamp[val_size:, :]
batch_size = 32
train_loader, x_train, y_train, x_train_mark, y_train_mark = tslib_data_loader(window, length_size, batch_size,
                                                                               data_train, data_train_mark)
val_loader, x_val, y_val, x_val_mark, y_val_mark = tslib_data_loader(window, length_size, batch_size, data_val,
                                                                     data_val_mark)
test_loader, x_test, y_test, x_test_mark, y_test_mark = tslib_data_loader(window, length_size, batch_size, data_test,
                                                                          data_test_mark)

                                                                          """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 100  # 训练迭代次数
learning_rate = 0.0001  # 学习率
scheduler_patience = int(0.25 * num_epochs)  # 转换为整数  学习率调整的patience
early_patience = 0.2  # 训练迭代的早停比例 即patience=0.25*num_epochs


class Config:
    def __init__(self):
        # basic
        self.seq_len = window  # 输入序列长度
        self.label_len = int(window / 2)  # 标签序列长度
        self.pred_len = length_size  # 预测序列长度
        #self.freq = 'b'  # 时间的频率，
        self.freq = 'h'  # 切换为小时级别频率，适配半小时高频数据
        # 模型训练
        self.batch_size = batch_size  # 批次大小
        self.num_epochs = num_epochs  # 训练的轮数
        self.learning_rate = learning_rate  # 学习率
        self.stop_ratio = early_patience  # 早停的比例
        # 模型 define
        self.dec_in = data_dim  # 解码器输入特征数量, 输入几个变量就是几
        self.enc_in = data_dim  # 编码器输入特征数量
        self.c_out = 1  # 输出维度##########这个很重要
        # 模型超参数
        self.d_model = 64  # 模型维度
        self.n_heads = 4  # 多头注意力头数
        self.dropout = 0.1  # 丢弃率
        self.e_layers = 3  # 编码器块的数量
        self.d_layers = 3  # 解码器块的数量
        self.d_ff = 128  # 全连接网络维度
        self.factor = 5  # 注意力因子
        self.activation = 'gelu'  # 激活函数
        self.channel_independence = 0  # 频道独立性，0:频道依赖，1:频道独立

        self.top_k = 6  # TimesBlock中的参数
        self.num_kernels = 6  # Inception中的参数
        self.distil = 1  # 是否使用蒸馏，1为True
        # 一般不需要动的参数
        self.embed = 'timeF'  # 时间特征编码方式
        self.output_attention = 0  # 是否输出注意力
        self.task_name = 'short_term_forecast'  # 模型的任务，一般不动但是必须这个参数


config = Config()

model_type = 'DCATCN-TCNInformer'
net = TCNInformer.Model(config).to(device)

criterion = nn.MSELoss().to(device)  # 损失函数 (回归 MSE 以保证波峰捕捉力度)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # 优化器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=scheduler_patience)  # 学习率调整策略

trained_model, train_loss, final_epoch = model_train(net, train_loader, length_size, optimizer, criterion, num_epochs,
                                                     device, print_train=True)
"""
trained_model, train_loss, val_loss, final_epoch = model_train_val(
    net=net,
    train_loader=train_loader,
    val_loader=val_loader,
    length_size=length_size,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    num_epochs=num_epochs,
    device=device,
    early_patience=early_patience,
    print_train=False       
)
"""
    #print_train改为True就会显示轮次，而不是早停法，用False就是早停法

trained_model.eval()  # 模型转换为验证模式
# 预测并调整维度
preds = []
trues = []
with torch.no_grad():
    for x, y, x_mark, y_mark in test_loader:
        x, y, x_mark, y_mark = x.to(device), y.to(device), x_mark.to(device), y_mark.to(device)
        outputs = trained_model(x, x_mark, y, y_mark)
        preds.append(outputs.detach().cpu().numpy())
        trues.append(y[:, -length_size:, -1:].detach().cpu().numpy())

pred = np.concatenate(preds, axis=0)
true = np.concatenate(trues, axis=0)

# 检查pred和true的维度并调整
print("Shape of true before adjustment:", true.shape)
print("Shape of pred before adjustment:", pred.shape)

# 可能需要调整pred和true的维度，使其变为二维数组
true = true[:, :, -1]
pred = pred[:, :, -1]  # 假设需要将pred调整为二维数组，去掉最后一维
# true =np.array(true)
# 假设需要将true调整为二维数组

print("Shape of pred after adjustment:", pred.shape)
print("Shape of true after adjustment:", true.shape)

# 这段代码是为了重新更新scaler，因为之前定义的scaler是是十六维，这里重新根据目标数据定义一下scaler
y_data_test_inverse = scaler.fit_transform(np.array(data_target).reshape(-1, 1))
pred_uninverse = scaler.inverse_transform(pred[:, -1:])  # 如果是多步预测， 选取最后一列
true_uninverse = scaler.inverse_transform(true[:, -1:])

true, pred = true_uninverse, pred_uninverse

df_eval = cal_eval(true, pred)  # 评估指标dataframe
print(df_eval)

# --- 保存结果 ---
# 创建 result 和 img 文件夹
output_dir = 'result'
img_dir = os.path.join(output_dir, 'img')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

# 1. 提取数据集名称和当前时间戳
dataset_name = os.path.splitext(os.path.basename(data_path))[0]
now = datetime.now().strftime("%Y%m%d_%H%M%S")

# 2. 定义本次运行的专属文件夹名称 (例如: TCNInformer_20231024_153000_DT_NEE)
run_folder_name = f"TCNInformer_{now}_{dataset_name}"

# 3. 创建专属文件夹路径 (result/TCNInformer_20231024_153000_DT_NEE)
output_dir = os.path.join('result', run_folder_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"\n==========================================")
print(f"[INFO] 本次运行的所有结果将保存在: {output_dir}")
print(f"==========================================")

# 4. 保存评估指标 (Metrics) 到专属文件夹
metrics_filename = f'{run_folder_name}_metrics.csv'
metrics_path = os.path.join(output_dir, metrics_filename)
df_eval.to_csv(metrics_path, index=False, encoding='utf-8-sig')
print(f'[SUCCESS] 评估指标已保存: {metrics_filename}')

# 5. 保存真实值和预测值 (Data) 到专属文件夹
test_dates = df['date'].iloc[-len(true.flatten()):].reset_index(drop=True)
data_filename = f'{run_folder_name}_data.csv'
data_path = os.path.join(output_dir, data_filename)
result_df = pd.DataFrame({'时间': test_dates,'真实值': true.flatten(), '预测值': pred.flatten()})
result_df.to_csv(data_path, index=False, encoding='utf-8-sig')
print(f'[SUCCESS] 预测数据已保存: {data_filename}')

# 6. 绘制并保存结果图 (Image) 到专属文件夹
df_pred_true = pd.DataFrame({'Predict': pred.flatten(), 'Real': true.flatten()})
plt.figure(figsize=(12, 4))
plt.plot(df_pred_true['Predict'], label='Predict', color='red', alpha=0.8)
plt.plot(df_pred_true['Real'], label='Real', color='blue', alpha=0.5)

# 标题带上数据集名称
plt.title(f'{model_type} Result ({dataset_name})')
plt.legend()

# 保存图片
img_filename = f'{run_folder_name}.png'
img_save_path = os.path.join(output_dir, img_filename)
plt.savefig(img_save_path, bbox_inches='tight')
print(f'[SUCCESS] 预测结果图已保存: {img_filename}\n')

# 最后展示图片
plt.show()
