from pathlib import Path
import pandas as pd
import numpy as np

def LoadData(envidir, fluxdir):
    """
    加载并处理环境与通量数据，返回清理后的数据集。

    参数：
    envidir (str): 环境数据文件路径。
    fluxdir (str): 通量数据文件路径。

    返回：
    pd.DataFrame: 处理后的数据集
    """

    envidata = pd.read_csv(envidir, header=0)
    rawdata = pd.read_csv(fluxdir, header=1)

    #删除单位行
    envidata = envidata.drop(0).reset_index(drop=True, inplace=False)
    rawdata = rawdata.drop(0).reset_index(drop=True, inplace=False)
    rawdata['datetime'] = rawdata['date'].astype(str) + ' ' + rawdata['time'].astype(str) #新建datetime列
    DOY = rawdata['DOY']
    #合并为一个表
    rawdata = pd.concat([envidata, rawdata], axis=1)

    # 设定需要保持原数据类型的列名
    columns_to_keep = ['datetime', "date", "time", "filename"]
    # 更改除column_to_keep之外的所有列的数据类型为float
    for col in rawdata.columns:
        if col not in columns_to_keep:
            rawdata[col] = rawdata[col].astype(float)

    #两个传感器的，一个缺失了另一个补
    rawdata.loc[rawdata['TA_1_1_1'].eq(-9999), 'TA_1_1_1'] = rawdata.loc[rawdata['TA_1_1_1'].eq(-9999),'air_temperature']
    rawdata.loc[rawdata['RH_1_1_1'].eq(-9999), 'RH_1_1_1'] = rawdata.loc[rawdata['RH_1_1_1'].eq(-9999),'RH']

    #计算NEE
    rawdata['NEE'] = rawdata['co2_flux'] + rawdata['co2_strg']
    rawdata.loc[rawdata['co2_flux'].eq(-9999), 'NEE'] = np.nan  #剔除因'co2_flux'值为-9999产生的错误NEE值,下同
    rawdata.loc[rawdata['co2_strg'].eq(-9999), 'NEE'] = np.nan  
    rawdata.loc[rawdata['qc_co2_flux'].eq(-9999), 'NEE'] = np.nan   
    #计算Rn
    rawdata['Rn'] = rawdata['SWIN_1_1_1'] + rawdata['LWIN_1_1_1'] - rawdata['SWOUT_1_1_1'] - rawdata['LWOUT_1_1_1']
    rawdata.loc[rawdata['LWIN_1_1_1'].eq(-9999), 'Rn'] = np.nan  
    rawdata.loc[rawdata['SWIN_1_1_1'].eq(-9999), 'Rn'] = np.nan  
    rawdata.loc[rawdata['SWOUT_1_1_1'].eq(-9999), 'Rn'] = np.nan  
    rawdata.loc[rawdata['LWOUT_1_1_1'].eq(-9999), 'Rn'] = np.nan  
    #将温度单位改为摄氏度
    rawdata['Tair'] = rawdata['TA_1_1_1'] - 273.15
    rawdata.loc[rawdata['TA_1_1_1'].eq(-9999), 'Tair'] = np.nan 
    rawdata['Tsoil'] = rawdata['TS_1_1_1'] - 273.15
    rawdata.loc[rawdata['TS_1_1_1'].eq(-9999), 'Tsoil'] = np.nan 

    #剔除QC=2的
    rawdata.loc[rawdata['qc_H'].eq(2), 'H'] = np.nan  
    rawdata.loc[rawdata['qc_LE'].eq(2), 'LE'] = np.nan  
    rawdata.loc[rawdata['qc_co2_flux'].eq(2), 'co2_flux'] = np.nan  
    rawdata.loc[rawdata['qc_co2_flux'].eq(2), 'NEE'] = np.nan  

    #提取后面要用的列
    col_names = ["datetime", 'Tair', "Tsoil", "SWC_1_1_1", "SWIN_1_1_1", "PPFD_1_1_1", "RH_1_1_1", "air_pressure", "wind_speed", "wind_dir", "VPD", "Rn", "H", "LE", "NEE", "P_RAIN_1_1_1", "SHF_1_1_1", "u*"] # 要提取的列名列表
    data = rawdata[col_names] # 使用列名列表提取多个指定列
    data.insert(1, 'DOY', DOY) #加入DOY列

    #将data中的-9999改为空值
    data.replace(-9999, np.nan, inplace=True)
    return data


def LoadData_nobiomet(fluxdir):
    """
    加载并处理通量数据，返回清理后的数据集。

    参数：
    fluxdir (str): 通量数据文件路径。

    返回：
    pd.DataFrame: 处理后的数据集
    """

    rawdata = pd.read_csv(fluxdir, header=1)

    #删除单位行
    rawdata = rawdata.drop(0).reset_index(drop=True, inplace=False)
    rawdata['datetime'] = rawdata['date'].astype(str) + ' ' + rawdata['time'].astype(str) #新建datetime列
    DOY = rawdata['DOY']

    # 设定需要保持原数据类型的列名
    columns_to_keep = ['datetime', "date", "time", "filename"]
    # 更改除column_to_keep之外的所有列的数据类型为float
    for col in rawdata.columns:
        if col not in columns_to_keep:
            rawdata[col] = rawdata[col].astype(float)

    #计算NEE
    rawdata['NEE'] = rawdata['co2_flux'] + rawdata['co2_strg']
    rawdata.loc[rawdata['co2_flux'].eq(-9999), 'NEE'] = np.nan  #剔除因'co2_flux'值为-9999产生的错误NEE值,下同
    rawdata.loc[rawdata['co2_strg'].eq(-9999), 'NEE'] = np.nan  
    rawdata.loc[rawdata['qc_co2_flux'].eq(-9999), 'NEE'] = np.nan   

    #将温度单位改为摄氏度
    # rawdata.loc[rawdata['air_temperature'].eq(-9999)] = np.nan 
    rawdata['Tair'] = rawdata['air_temperature']
    rawdata['Tair'] = rawdata['Tair'].where(rawdata['Tair'] == -9999, rawdata['Tair'] - 273.15)
    # rawdata['Tsoil'] = rawdata['TS_1_1_1'] - 273.15 #无Tsoil
    # rawdata.loc[rawdata['TS_1_1_1'].eq(-9999), 'Tsoil'] = np.nan 

    #剔除QC=2的
    rawdata.loc[rawdata['qc_H'].eq(2), 'H'] = np.nan  
    rawdata.loc[rawdata['qc_LE'].eq(2), 'LE'] = np.nan  
    rawdata.loc[rawdata['qc_co2_flux'].eq(2), 'co2_flux'] = np.nan  
    rawdata.loc[rawdata['qc_co2_flux'].eq(2), 'NEE'] = np.nan  

    #提取后面要用的列
    col_names = ["datetime", 'Tair', "air_pressure", "wind_speed", "wind_dir", "RH", "VPD", "H", "LE", "ET", "NEE", "u*"] # 要提取的列名列表
    data = rawdata[col_names] # 使用列名列表提取多个指定列
    data.insert(1, 'DOY', DOY) #加入DOY列

    #将data中的-9999改为空值
    data.replace(-9999, np.nan, inplace=True)
    return data

if __name__ == "__main__":
    #先获取我py文件的路径 -- 当前绝对路径
    codepath = Path(__file__).absolute()
    current_path = str(codepath.parent.parent) 

    enviFilename = "eddypro_240602_biomet_2024-06-13T125920_exp.csv" #biomet
    fluxFilename = "eddypro_240602_full_output_2024-06-13T125920_exp.csv" #full_output
    envidir = current_path + "\\" + enviFilename
    fluxdir = current_path + "\\" + fluxFilename

    data = LoadData(envidir, fluxdir)
    data.to_csv(current_path + "\\proccess_demo_DBS\\" + 'DBS_L2_24_1.csv', index=False, mode="w")