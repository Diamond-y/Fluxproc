import pandas as pd
import numpy as np
import math
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt; plt.style.use('seaborn-v0_8')

def BuildDataset(rawfile, ERA5file=None):
    if ERA5file is None:
        rawdata = pd.read_csv(rawfile, header=0)
        #重命名列(如果列不存在, 将被忽略)
        rawdata.rename(columns={'SWC_1_1_1':'SWC', 'SWIN_1_1_1':'SWIN', 'PPFD_1_1_1':'PPFD', 'RH_1_1_1':'RH', 'SHF_1_1_1':'SHF', 'P_RAIN_1_1_1':'pr', 'air_pressure':'PA', "wind_speed":"WS", "wind_dir":"WD"}, inplace=True)
        rawdata['datetime'] = pd.to_datetime(rawdata['datetime'])  # 确保 datetime 列是时间格式
        rawdata = rawdata.sort_values(by='datetime')  # 按时间排序
        #提取后面要用的列(如果列不存在, 将被忽略)
        col_names = ["datetime", 'DOY', 'Tair', "Tsoil", "SWC", "SWIN", "PPFD", "RH", "ET", "VPD", "Rn", "Rg", "pr", "PA", "WS", "WD", "H", "LE", "NEE", "SHF", "u*",
                    'Tair_era5', 'Tsoil_era5', 'SWC_era5', 'SWIN_era5', 'RH_era5', 'VPD_era5', 'Rn_era5', 'pr_era5'] # 要提取的列名列表
        exsist_columns = {col: rawdata[col] for col in col_names if col in rawdata.columns} # 使用字典推导式提取存在的列
        dataset = pd.DataFrame(exsist_columns) # 将字典转换为DataFrame
        return dataset
    else:
        ERA5data = pd.read_csv(ERA5file, header=0)
        rawdata = pd.read_csv(rawfile, header=0)
        #重命名列(如果列不存在, 将被忽略)
        ERA5data.rename(columns={'Fn':'Rn_era5', 'Fsd':'SWIN_era5', 'RH':'RH_era5', 'Sws':'SWC_era5', 'Ta':'Tair_era5', 'Ts':'Tsoil_era5', 'VPD':'VPD_era5'}, inplace=True)
        rawdata.rename(columns={'SWC_1_1_1':'SWC', 'SWIN_1_1_1':'SWIN', 'PPFD_1_1_1':'PPFD', 'RH_1_1_1':'RH', 'SHF_1_1_1':'SHF', 'P_RAIN_1_1_1':'pr', 'air_pressure':'PA', "wind_speed":"WS", "wind_dir":"WD"}, inplace=True)
        ERA5data['datetime'] = pd.to_datetime(ERA5data['datetime'])  # 确保 datetime 列是时间格式
        ERA5data = ERA5data.sort_values(by='datetime')  # 按时间排序
        rawdata['datetime'] = pd.to_datetime(rawdata['datetime'])  # 确保 datetime 列是时间格式
        rawdata = rawdata.sort_values(by='datetime')  # 按时间排序

        # alldata = pd.concat([rawdata, ERA5data], axis=1) #必须保证时间对应才能做！
        alldata = pd.merge(rawdata, ERA5data, on='datetime', how='inner')

        #提取后面要用的列(如果列不存在, 将被忽略)
        col_names = ["datetime", 'DOY', 'Tair', "Tsoil", "SWC", "SWIN", "PPFD", "RH", "ET", "VPD", "Rn", "Rg", "pr", "PA", "WS", "WD", "H", "LE", "NEE", "SHF", "u*",
                    'Tair_era5', 'Tsoil_era5', 'SWC_era5', 'SWIN_era5', 'RH_era5', 'VPD_era5', 'Rn_era5', 'pr_era5'] # 要提取的列名列表
        exsist_columns = {col: alldata[col] for col in col_names if col in alldata.columns} # 使用字典推导式提取存在的列
        dataset = pd.DataFrame(exsist_columns) # 将字典转换为DataFrame
        return dataset

def XGboostGapFilling(dataset, var_to_fill, Mode='use_era5_only', X_col=[], pred_all=True,    
    max_depth=3,learning_rate=0.1,n_estimators=100,booster='gbtree',gamma=0,min_child_weight=1,subsample=1,colsample_bytree=1,reg_alpha=0,reg_lambda=1,random_state=0):
    #划分数据集
    train_data = dataset[dataset[var_to_fill].notnull()]
    pred_data = dataset[dataset[var_to_fill].isnull()]
    pred_data.drop(var_to_fill, axis=1, inplace=True)
    datetime = pred_data['datetime']

    if Mode == 'use_era5_only':
        X_col = ['DOY', 'Tair_era5', 'Tsoil_era5', 'SWC_era5', 'SWIN_era5', 'RH_era5', 'VPD_era5', 'Rn_era5']
    elif Mode == 'use_custom_X_col':
        X_col = X_col
    else:
        X_col = dataset.columns.tolist()
        columns_to_remove = ['datetime', 'LE', 'NEE', 'H', 'SHF', 'u*']
        X_col = [col for col in X_col if col not in columns_to_remove]
        if var_to_fill in X_col:
            X_col.remove(var_to_fill)
        else:
            print('No variable names ' + var_to_fill + ',skip this variable')
            dataset[var_to_fill + '_XGBGapFilled'] = None  
            return dataset
        
    X = train_data[X_col]
    y = train_data[[var_to_fill]].squeeze()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) #30%做测试集

    #训练xgboost模型
    model = xgb.XGBRegressor(max_depth=max_depth,          # 每一棵树最大深度, 默认6；
                        learning_rate=learning_rate,      # 学习率, 每棵树的预测结果都要乘以这个学习率, 默认0.3；
                        n_estimators=n_estimators,        # 使用多少棵树来拟合, 也可以理解为多少次迭代。默认100；
                        booster=booster,         # 有两种模型可以选择gbtree和gblinear。gbtree使用基于树的模型进行提升计算, gblinear使用线性模型进行提升计算。默认为gbtree
                        gamma=gamma,                 # 叶节点上进行进一步分裂所需的最小"损失减少"。默认0；
                        min_child_weight=min_child_weight,      # 可以理解为叶子节点最小样本数, 默认1；
                        subsample=subsample,              # 训练集抽样比例, 每次拟合一棵树之前, 都会进行该抽样步骤。默认1, 取值范围(0, 1]
                        colsample_bytree=colsample_bytree,       # 每次拟合一棵树之前, 决定使用多少个特征, 参数默认1, 取值范围(0, 1]。
                        reg_alpha=reg_alpha,             # 默认为0, 控制模型复杂程度的权重值的 L1 正则项参数, 参数值越大, 模型越不容易过拟合。
                        reg_lambda=reg_lambda,            # 默认为1, 控制模型复杂度的权重值的L2正则化项参数, 参数越大, 模型越不容易过拟合。
                        random_state=random_state)        # 随机种子
    model.fit(X_train, y_train)
    #检验
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    RMSE_train = math.sqrt(mean_squared_error(y_true=y_train, y_pred=y_train_pred))
    RMSE_test = math.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    MAE_train = mean_absolute_error(y_true=y_train, y_pred=y_train_pred)
    MAE_test = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)
    r2_train = r2_score(y_true=y_train, y_pred=y_train_pred)
    r2_test = r2_score(y_true=y_test, y_pred=y_test_pred)
    print("Filled_var: " + var_to_fill + '\n' + "RMSE_train: " + str(RMSE_train) + ' ' + "RMSE_test: " + str(RMSE_test) + '\n' + "MAE_train: " + str(MAE_train)
          + ' ' + "MAE_test: " + str(MAE_test) + '\n' + "r2_train: " + str(r2_train) + ' ' + "r2_test: " + str(r2_test))
    #预测
    y_pred = model.predict(pred_data[X_col])
    y_pred_df = pd.DataFrame(y_pred, columns=[var_to_fill+'_XGBpred'])
    if var_to_fill in ["SWC", "SWIN", "PPFD", "RH", "VPD", "Rn", "Rg", "pr"]: #将小于0预测值的变为0
        y_pred_df[var_to_fill+'_XGBpred'] = y_pred_df[var_to_fill+'_XGBpred'].apply(lambda x: 0 if x < 0 else x)
    y_pred_df['datetime'] = datetime.values
    data_pred = pd.merge(dataset, y_pred_df, on='datetime', how='outer')
    data_pred['datetime'] = pd.to_datetime(data_pred['datetime'])  # 转换为时间类型
    data_pred = data_pred.sort_values(by='datetime')  # 时间顺序排序
    data_pred[var_to_fill + '_XGBGapFilled'] = data_pred.apply(
        lambda row: row[var_to_fill + '_XGBpred'] if pd.notnull(row[var_to_fill + '_XGBpred']) else row[var_to_fill], axis=1)
    # data_pred.to_csv(r"E:\term1\flux_postprocessing\proccess_demo_DBS\data_pred.csv", index=False, mode="w")

    ###画图
    #feature importance
    plt.figure(figsize=(15,5))
    plt.bar(range(len(X_col)), model.feature_importances_)
    plt.xticks(range(len(X_col)), X_col)
    plt.title('feature importance')
    plt.show()
    #画插补结果折线图
    plt.figure(figsize=(20,6))
    plt.scatter(data_pred['datetime'], data_pred[var_to_fill], label=var_to_fill, color='blue', s=15)
    plt.scatter(data_pred['datetime'], data_pred[var_to_fill+'_XGBpred'], label=var_to_fill+'_XGBpred', color='red', s=20)
    plt.title('XGboost Gapfill Result: '+var_to_fill+f"  |  RMSE_test:{RMSE_test:.2f}  MAE_test:{MAE_test:.2f}  r2_test:{r2_test:.2f}", fontsize=22)
    # 确定目标 x 轴标签数量
    xtick_num = 45  # 目标显示的标签数量
    step = max(1, len(data_pred['datetime']) // xtick_num)  # 根据目标数量计算步长
    # 设置 x 轴标签
    plt.xticks(data_pred['datetime'][::step], rotation=45)
    # plt.text(900,1750, "RMSE_test:{:.2f}  MAE_test:{:.2f}  r2_test:{:.2f}".format(RMSE_test, MAE_test, r2_test), fontsize=16)
    plt.ylabel(var_to_fill, fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if pred_all:
        #预测全部
        y_pred_all = model.predict(dataset[X_col])
        y_pred_all_df = pd.DataFrame(y_pred_all, columns=[var_to_fill+'_XGBpred'])
        y_pred_all_df['datetime'] = dataset['datetime'].values
        data_pred_all = pd.merge(dataset, y_pred_all_df, on='datetime', how='outer')
        data_pred_all['datetime'] = pd.to_datetime(data_pred_all['datetime'])  # 转换为时间类型
        data_pred_all = data_pred_all.sort_values(by='datetime')  # 时间顺序排序
        #画预测全部结果折线图
        plt.figure(figsize=(20,6))
        plt.scatter(data_pred_all['datetime'], data_pred_all[var_to_fill], label=var_to_fill, color='blue', s=15)
        plt.scatter(data_pred_all['datetime'], data_pred_all[var_to_fill+'_XGBpred']*2, label=var_to_fill+'_XGBpred*2', color='red', s=20)
        plt.title('XGBboost perdict all: '+var_to_fill, fontsize=22)
        # 确定目标 x 轴标签数量
        xtick_num = 45  # 目标显示的标签数量
        step = max(1, len(data_pred_all['datetime']) // xtick_num)  # 根据目标数量计算步长
        # 设置 x 轴标签
        plt.xticks(data_pred_all['datetime'][::step], rotation=45)
        # plt.text(900,1750, "RMSE_test:{:.2f}  MAE_test:{:.2f}  r2_test:{:.2f}".format(RMSE_test, MAE_test, r2_test), fontsize=16)
        plt.ylabel(var_to_fill, fontsize=16)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return data_pred

def RandomForestGapFilling(dataset, var_to_fill, Mode='use_era5_only', X_col=[], pred_all=True,   
    n_estimators=10,criterion='squared_error',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0,max_features=None,max_leaf_nodes=None,bootstrap=True,oob_score=False,n_jobs=1,random_state=None,verbose=0,warm_start=False):
    from sklearn.ensemble import RandomForestRegressor

    #划分数据集
    train_data = dataset[dataset[var_to_fill].notnull()]
    pred_data = dataset[dataset[var_to_fill].isnull()]
    pred_data.drop(var_to_fill, axis=1, inplace=True)
    datetime = pred_data['datetime']

    if Mode == 'use_era5_only':
        X_col = ['DOY', 'Tair_era5', 'Tsoil_era5', 'SWC_era5', 'SWIN_era5', 'RH_era5', 'VPD_era5', 'Rn_era5']
    elif Mode == 'use_custom_X_col':
        X_col = X_col
    else:
        X_col = dataset.columns.tolist()
        columns_to_remove = ['datetime', 'LE', 'NEE', 'H', 'SHF', 'u*']
        X_col = [col for col in X_col if col not in columns_to_remove]
        if var_to_fill in X_col:
            X_col.remove(var_to_fill)
        else:
            print('No variable names ' + var_to_fill + ',skip this variable')
            dataset[var_to_fill + '_RFGapFilled'] = None  
            return dataset
        
    X = train_data[X_col]
    y = train_data[[var_to_fill]].squeeze()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) #30%做测试集

    #训练模型
    model = RandomForestRegressor( n_estimators=n_estimators,   #  数值型参数, 默认值为100, 此参数指定了弱分类器的个数。设置的值越大, 精确度越好, 但是当 n_estimators 大于特定值之后, 带来的提升效果非常有限。
                                criterion=criterion,   # 其中, 参数criterion 是字符串类型, 可选的有'squared_error', 'poisson', 'absolute_error', 'friedman_mse'。
                                max_depth=max_depth,    # 数值型, 默认值None。这是与剪枝相关的参数, 设置为None时, 树的节点会一直分裂, 直到：（1）每个叶子都是“纯”的；（2）或者叶子中包含于min_sanples_split个样本。推荐从 max_depth = 3 尝试增加, 观察是否应该继续加大深度。
                                min_samples_split=min_samples_split,  # 数值型, 默认值2, 指定每个内部节点(非叶子节点)包含的最少的样本数。与min_samples_leaf这个参数类似, 可以是整数也可以是浮点数。
                                min_samples_leaf=min_samples_leaf,  # 数值型, 默认值1, 指定每个叶子结点包含的最少的样本数。参数的取值除了整数之外, 还可以是浮点数, 此时（min_samples_leaf * n_samples）向下取整后的整数是每个节点的最小样本数。此参数设置的过小会导致过拟合, 反之就会欠拟合。
                                min_weight_fraction_leaf=min_weight_fraction_leaf,  # (default=0) 叶子节点所需要的最小权值
                                max_features=max_features,   # 可以为整数、浮点、字符或者None, 默认值为None。此参数用于限制分枝时考虑的特征个数, 超过限制个数的特征都会被舍弃。
                                max_leaf_nodes=max_leaf_nodes,   # 数值型参数, 默认值为None, 即不限制最大叶子节点数。这个参数通过限制树的最大叶子数量来防止过拟合, 如果设置了一个正整数, 则会在建立的最大叶节点内的树中选择最优的决策树。
                                bootstrap=bootstrap,        # 是否有放回的采样。
                                oob_score=oob_score,       #  oob（out of band, 带外）数据, 即：在某次决策树训练中没有被bootstrap选中的数据
                                n_jobs=n_jobs,              # 并行job个数。
                                random_state=random_state,      # 随机种子
                                verbose=verbose,          # (default=0) 是否显示任务进程
                                warm_start=warm_start)   # 热启动, 决定是否使用上次调用该类的结果然后增加新的。
    model.fit(X_train, y_train)
    #检验
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    RMSE_train = math.sqrt(mean_squared_error(y_true=y_train, y_pred=y_train_pred))
    RMSE_test = math.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    MAE_train = mean_absolute_error(y_true=y_train, y_pred=y_train_pred)
    MAE_test = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)
    r2_train = r2_score(y_true=y_train, y_pred=y_train_pred)
    r2_test = r2_score(y_true=y_test, y_pred=y_test_pred)
    print("Filled_var: " + var_to_fill + '\n' + "RMSE_train: " + str(RMSE_train) + ' ' + "RMSE_test: " + str(RMSE_test) + '\n' + "MAE_train: " + str(MAE_train)
          + ' ' + "MAE_test: " + str(MAE_test) + '\n' + "r2_train: " + str(r2_train) + ' ' + "r2_test: " + str(r2_test))
    #预测
    y_pred = model.predict(pred_data[X_col])
    y_pred_df = pd.DataFrame(y_pred, columns=[var_to_fill+'_RFpred'])
    y_pred_df['datetime'] = datetime.values
    data_pred = pd.merge(dataset, y_pred_df, on='datetime', how='outer')
    data_pred['datetime'] = pd.to_datetime(data_pred['datetime'])  # 转换为时间类型
    data_pred = data_pred.sort_values(by='datetime')  # 时间顺序排序
    data_pred[var_to_fill + '_RFGapFilled'] = data_pred.apply(
        lambda row: row[var_to_fill + '_RFpred'] if pd.notnull(row[var_to_fill + '_RFpred']) else row[var_to_fill], axis=1)
    # data_pred.to_csv(r"E:\term1\flux_postprocessing\proccess_demo_DBS\data_pred.csv", index=False, mode="w")

    ###画图
    #feature importance
    plt.figure(figsize=(15,5))
    plt.bar(range(len(X_col)), model.feature_importances_)
    plt.xticks(range(len(X_col)), X_col)
    plt.title('feature importance')
    plt.show()
    #画插补结果折线图
    plt.figure(figsize=(20,6))
    plt.scatter(data_pred['datetime'], data_pred[var_to_fill], label=var_to_fill, color='blue', s=15)
    plt.scatter(data_pred['datetime'], data_pred[var_to_fill+'_RFpred'], label=var_to_fill+'_RFpred', color='red', s=20)
    plt.title('RF Gapfill Result: '+var_to_fill+f"  |  RMSE_test:{RMSE_test:.2f}  MAE_test:{MAE_test:.2f}  r2_test:{r2_test:.2f}", fontsize=22)
    # 确定目标 x 轴标签数量
    xtick_num = 45  # 目标显示的标签数量
    step = max(1, len(data_pred['datetime']) // xtick_num)  # 根据目标数量计算步长
    # 设置 x 轴标签
    plt.xticks(data_pred['datetime'][::step], rotation=45)
    # plt.text(900,1750, "RMSE_test:{:.2f}  MAE_test:{:.2f}  r2_test:{:.2f}".format(RMSE_test, MAE_test, r2_test), fontsize=16)
    plt.ylabel(var_to_fill, fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if pred_all:
        #预测全部
        y_pred_all = model.predict(dataset[X_col])
        y_pred_all_df = pd.DataFrame(y_pred_all, columns=[var_to_fill+'_RFpred'])
        y_pred_all_df['datetime'] = dataset['datetime'].values
        data_pred_all = pd.merge(dataset, y_pred_all_df, on='datetime', how='outer')
        data_pred_all['datetime'] = pd.to_datetime(data_pred_all['datetime'])  # 转换为时间类型
        data_pred_all = data_pred_all.sort_values(by='datetime')  # 时间顺序排序
        #画预测全部结果折线图
        plt.figure(figsize=(20,6))
        plt.scatter(data_pred_all['datetime'], data_pred_all[var_to_fill], label=var_to_fill, color='blue', s=15)
        plt.scatter(data_pred_all['datetime'], data_pred_all[var_to_fill+'_RFpred']*2, label=var_to_fill+'_RFpred*2', color='red', s=20)
        plt.title('RF perdict all: '+var_to_fill, fontsize=22)
        # 确定目标 x 轴标签数量
        xtick_num = 45  # 目标显示的标签数量
        step = max(1, len(data_pred_all['datetime']) // xtick_num)  # 根据目标数量计算步长
        # 设置 x 轴标签
        plt.xticks(data_pred_all['datetime'][::step], rotation=45)
        # plt.text(900,1750, "RMSE_test:{:.2f}  MAE_test:{:.2f}  r2_test:{:.2f}".format(RMSE_test, MAE_test, r2_test), fontsize=16)
        plt.ylabel(var_to_fill, fontsize=16)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return data_pred

def AdaboostGapFilling(dataset, var_to_fill, Mode='use_era5_only', X_col=[], pred_all=True,   
                       n_estimators=100, learning_rate=1.0,loss= "linear",random_state=0):
    from sklearn.ensemble import AdaBoostRegressor

    #划分数据集
    train_data = dataset[dataset[var_to_fill].notnull()]
    pred_data = dataset[dataset[var_to_fill].isnull()]
    pred_data.drop(var_to_fill, axis=1, inplace=True)
    datetime = pred_data['datetime']

    if Mode == 'use_era5_only':
        X_col = ['DOY', 'Tair_era5', 'Tsoil_era5', 'SWC_era5', 'SWIN_era5', 'RH_era5', 'VPD_era5', 'Rn_era5']
    elif Mode == 'use_custom_X_col':
        X_col = X_col
    else:
        X_col = dataset.columns.tolist()
        columns_to_remove = ['datetime', 'LE', 'NEE', 'H', 'SHF', 'u*']
        X_col = [col for col in X_col if col not in columns_to_remove]
        if var_to_fill in X_col:
            X_col.remove(var_to_fill)
        else:
            print('No variable names ' + var_to_fill + ',skip this variable')
            dataset[var_to_fill + '_AdaGapFilled'] = None  
            return dataset
        
    X = train_data[X_col]
    y = train_data[[var_to_fill]].squeeze()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) #30%做测试集

    #训练模型
    model = AdaBoostRegressor( n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            loss=loss,
                            random_state=random_state)
    model.fit(X_train, y_train)
    #检验
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    RMSE_train = math.sqrt(mean_squared_error(y_true=y_train, y_pred=y_train_pred))
    RMSE_test = math.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    MAE_train = mean_absolute_error(y_true=y_train, y_pred=y_train_pred)
    MAE_test = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)
    r2_train = r2_score(y_true=y_train, y_pred=y_train_pred)
    r2_test = r2_score(y_true=y_test, y_pred=y_test_pred)
    print("Filled_var: " + var_to_fill + '\n' + "RMSE_train: " + str(RMSE_train) + ' ' + "RMSE_test: " + str(RMSE_test) + '\n' + "MAE_train: " + str(MAE_train)
          + ' ' + "MAE_test: " + str(MAE_test) + '\n' + "r2_train: " + str(r2_train) + ' ' + "r2_test: " + str(r2_test))
    #预测
    y_pred = model.predict(pred_data[X_col])
    y_pred_df = pd.DataFrame(y_pred, columns=[var_to_fill+'_Adapred'])
    y_pred_df['datetime'] = datetime.values
    data_pred = pd.merge(dataset, y_pred_df, on='datetime', how='outer')
    data_pred['datetime'] = pd.to_datetime(data_pred['datetime'])  # 转换为时间类型
    data_pred = data_pred.sort_values(by='datetime')  # 时间顺序排序
    data_pred[var_to_fill + '_AdaGapFilled'] = data_pred.apply(
        lambda row: row[var_to_fill + '_Adapred'] if pd.notnull(row[var_to_fill + '_Adapred']) else row[var_to_fill], axis=1)
    # data_pred.to_csv(r"E:\term1\flux_postprocessing\proccess_demo_DBS\data_pred.csv", index=False, mode="w")

    ###画图
    #feature importance
    plt.figure(figsize=(15,5))
    plt.bar(range(len(X_col)), model.feature_importances_)
    plt.xticks(range(len(X_col)), X_col)
    plt.title('feature importance')
    plt.show()
    #画插补结果折线图
    plt.figure(figsize=(20,6))
    plt.scatter(data_pred['datetime'], data_pred[var_to_fill], label=var_to_fill, color='blue', s=15)
    plt.scatter(data_pred['datetime'], data_pred[var_to_fill+'_Adapred'], label=var_to_fill+'_Adapred', color='red', s=20)
    plt.title('Adaboost Gapfill Result: '+var_to_fill, fontsize=22)
    # 确定目标 x 轴标签数量
    xtick_num = 45  # 目标显示的标签数量
    step = max(1, len(data_pred['datetime']) // xtick_num)  # 根据目标数量计算步长
    # 设置 x 轴标签
    plt.xticks(data_pred['datetime'][::step], rotation=45)
    # plt.text(900,1750, "RMSE_test:{:.2f}  MAE_test:{:.2f}  r2_test:{:.2f}".format(RMSE_test, MAE_test, r2_test), fontsize=16)
    plt.ylabel(var_to_fill, fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if pred_all:
        #预测全部
        y_pred_all = model.predict(dataset[X_col])
        y_pred_all_df = pd.DataFrame(y_pred_all, columns=[var_to_fill+'_Adapred'])
        y_pred_all_df['datetime'] = dataset['datetime'].values
        data_pred_all = pd.merge(dataset, y_pred_all_df, on='datetime', how='outer')
        data_pred_all['datetime'] = pd.to_datetime(data_pred_all['datetime'])  # 转换为时间类型
        data_pred_all = data_pred_all.sort_values(by='datetime')  # 时间顺序排序
        #画预测全部结果折线图
        plt.figure(figsize=(20,6))
        plt.scatter(data_pred_all['datetime'], data_pred_all[var_to_fill], label=var_to_fill, color='blue', s=15)
        plt.scatter(data_pred_all['datetime'], data_pred_all[var_to_fill+'_Adapred']*2, label=var_to_fill+'_Adapred*2', color='red', s=20)
        plt.title('Adaboost perdict all: '+var_to_fill+f"  |  RMSE_test:{RMSE_test:.2f}  MAE_test:{MAE_test:.2f}  r2_test:{r2_test:.2f}", fontsize=22)
        # 确定目标 x 轴标签数量
        xtick_num = 45  # 目标显示的标签数量
        step = max(1, len(data_pred_all['datetime']) // xtick_num)  # 根据目标数量计算步长
        # 设置 x 轴标签
        plt.xticks(data_pred_all['datetime'][::step], rotation=45)
        # plt.text(900,1750, "RMSE_test:{:.2f}  MAE_test:{:.2f}  r2_test:{:.2f}".format(RMSE_test, MAE_test, r2_test), fontsize=16)
        plt.ylabel(var_to_fill, fontsize=16)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return data_pred

def ANNGapFilling(dataset, var_to_fill, Mode='use_era5_only', X_col=[], pred_all=True,   
                  hidden_layer_sizes=(100,),activation='relu', *,solver='adam',alpha=0.0001,batch_size='auto',learning_rate='constant',learning_rate_init=0.001,power_t=0.5,max_iter=200,shuffle=True,tol=0.0001,verbose=False,warm_start=False,momentum=0.9,nesterovs_momentum=True,early_stopping=False,validation_fraction=0.1,beta_1=0.9,beta_2=0.999,epsilon=1e-08,n_iter_no_change=10,max_fun=15000,random_state=0):
    from sklearn.neural_network import MLPRegressor

    #划分数据集
    train_data = dataset[dataset[var_to_fill].notnull()]
    pred_data = dataset[dataset[var_to_fill].isnull()]
    pred_data.drop(var_to_fill, axis=1, inplace=True)
    datetime = pred_data['datetime']

    if Mode == 'use_era5_only':
        X_col = ['DOY', 'Tair_era5', 'Tsoil_era5', 'SWC_era5', 'SWIN_era5', 'RH_era5', 'VPD_era5', 'Rn_era5']
    elif Mode == 'use_custom_X_col':
        X_col = X_col
    else:
        X_col = dataset.columns.tolist()
        columns_to_remove = ['datetime', 'LE', 'NEE', 'H', 'SHF', 'u*']
        X_col = [col for col in X_col if col not in columns_to_remove]
        if var_to_fill in X_col:
            X_col.remove(var_to_fill)
        else:
            print('No variable names ' + var_to_fill + ',skip this variable')
            dataset[var_to_fill + '_ANNGapFilled'] = None  
            return dataset
    
    X = train_data[X_col]
    y = train_data[[var_to_fill]].squeeze()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) #30%做测试集

    #训练模型
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                         activation=activation,
                         solver=solver,
                         alpha=alpha,
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         learning_rate_init=learning_rate_init,
                         power_t=power_t,
                         max_iter=max_iter,
                         shuffle=shuffle,
                         tol=tol,
                         verbose=verbose,
                         warm_start=warm_start,
                         momentum=momentum,
                         nesterovs_momentum=nesterovs_momentum,
                         early_stopping=early_stopping,
                         validation_fraction=validation_fraction,
                         beta_1=beta_1,
                         beta_2=beta_2,
                         epsilon=epsilon,
                         n_iter_no_change=n_iter_no_change,
                         max_fun=max_fun,
                         random_state=random_state)
    model.fit(X_train, y_train)
    #检验
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    RMSE_train = math.sqrt(mean_squared_error(y_true=y_train, y_pred=y_train_pred))
    RMSE_test = math.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    MAE_train = mean_absolute_error(y_true=y_train, y_pred=y_train_pred)
    MAE_test = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)
    r2_train = r2_score(y_true=y_train, y_pred=y_train_pred)
    r2_test = r2_score(y_true=y_test, y_pred=y_test_pred)
    print("Filled_var: " + var_to_fill + '\n' + "RMSE_train: " + str(RMSE_train) + ' ' + "RMSE_test: " + str(RMSE_test) + '\n' + "MAE_train: " + str(MAE_train)
          + ' ' + "MAE_test: " + str(MAE_test) + '\n' + "r2_train: " + str(r2_train) + ' ' + "r2_test: " + str(r2_test))
 
    #预测缺失部分
    y_pred = model.predict(pred_data[X_col])
    y_pred_df = pd.DataFrame(y_pred, columns=[var_to_fill+'_ANNpred'])
    y_pred_df['datetime'] = datetime.values
    data_pred = pd.merge(dataset, y_pred_df, on='datetime', how='outer')
    data_pred['datetime'] = pd.to_datetime(data_pred['datetime'])  # 转换为时间类型
    data_pred = data_pred.sort_values(by='datetime')  # 时间顺序排序
    data_pred[var_to_fill + '_ANNGapFilled'] = data_pred.apply(
        lambda row: row[var_to_fill + '_ANNpred'] if pd.notnull(row[var_to_fill + '_ANNpred']) else row[var_to_fill], axis=1)
    # data_pred.to_csv(r"E:\term1\flux_postprocessing\proccess_demo_DBS\temp\data_pred_"+var_to_fill+".csv", index=False, mode="w")

    ###画图
    #画插补结果折线图
    plt.figure(figsize=(20,6))
    plt.scatter(data_pred['datetime'], data_pred[var_to_fill], label=var_to_fill, color='blue', s=15)
    plt.scatter(data_pred['datetime'], data_pred[var_to_fill+'_ANNpred'], label=var_to_fill+'_ANNpred', color='red', s=20)
    plt.title('ANN Gapfill Result: '+var_to_fill+f"  |  RMSE_test:{RMSE_test:.2f}  MAE_test:{MAE_test:.2f}  r2_test:{r2_test:.2f}", fontsize=22)
    # 确定目标 x 轴标签数量
    xtick_num = 45  # 目标显示的标签数量
    step = max(1, len(data_pred['datetime']) // xtick_num)  # 根据目标数量计算步长
    # 设置 x 轴标签
    plt.xticks(data_pred['datetime'][::step], rotation=45)
    # plt.text(900,1750, "RMSE_test:{:.2f}  MAE_test:{:.2f}  r2_test:{:.2f}".format(RMSE_test, MAE_test, r2_test), fontsize=16)
    plt.ylabel(var_to_fill, fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if pred_all:
        #预测全部
        y_pred_all = model.predict(dataset[X_col])
        y_pred_all_df = pd.DataFrame(y_pred_all, columns=[var_to_fill+'_ANNpred'])
        y_pred_all_df['datetime'] = dataset['datetime'].values
        data_pred_all = pd.merge(dataset, y_pred_all_df, on='datetime', how='outer')
        data_pred_all['datetime'] = pd.to_datetime(data_pred_all['datetime'])  # 转换为时间类型
        data_pred_all = data_pred_all.sort_values(by='datetime')  # 时间顺序排序
        #画预测全部结果折线图
        plt.figure(figsize=(20,6))
        plt.scatter(data_pred_all['datetime'], data_pred_all[var_to_fill], label=var_to_fill, color='blue', s=15)
        plt.scatter(data_pred_all['datetime'], data_pred_all[var_to_fill+'_ANNpred']*2, label=var_to_fill+'_ANNpred*2', color='red', s=20)
        plt.title('ANN perdict all: '+var_to_fill, fontsize=22)
        # 确定目标 x 轴标签数量
        xtick_num = 45  # 目标显示的标签数量
        step = max(1, len(data_pred_all['datetime']) // xtick_num)  # 根据目标数量计算步长
        # 设置 x 轴标签
        plt.xticks(data_pred_all['datetime'][::step], rotation=45)
        # plt.text(900,1750, "RMSE_test:{:.2f}  MAE_test:{:.2f}  r2_test:{:.2f}".format(RMSE_test, MAE_test, r2_test), fontsize=16)
        plt.ylabel(var_to_fill, fontsize=16)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return data_pred

def GapFillCols(dataset, gap_filling_method='XGB', num_to_fill_na=0, Mode='use_era5_only', vars_to_fill=[], X_col=[]):
    """
    选择一种方法, 一次对多列进行插补, 若要插补通量数据, 请令Mode='use_custom_X_col', 并指定vars_to_fill、X_col

    参数：
        dataset: pd.DataFrame, 数据集
        gap_filling_method: str, 插补方法(全称或简称, 无空格)
        num_to_fill_na: int, 使用有缺失的列做解释变量时, 用于填补空位的数, 默认0
        Mode: str, 使用哪些数据作为解释变量, 默认为除待插补变量外全部, 可选:'use_era5_only','use_custom_X_col'
        vars_to_fill: list, 需要插补的列的列名
        X_col: list, 可选, Mode='use_custom_X_col'时, 用作为解释变量的列的列名
    
    返回：
        dataset_GapFilled: pd.DataFrame, 新增插补后列的dataset
    """
    if vars_to_fill and X_col: #指定需插补的列和用于插补的变量, Mode必须为'use_custom_X_col'
        in_cols = vars_to_fill + X_col + ['datetime']
        in_cols = list(set(in_cols)) #去重
        mete_data = dataset.loc[:, in_cols]
        data_to_fill = dataset.loc[:, vars_to_fill]
        missing_values = data_to_fill.isnull().sum() # 计算需插补的列每列的缺失值数量
        sorted_vars_with_missing = missing_values[missing_values > 0].sort_values(ascending=True).index.tolist() # 筛选出含缺失值的列并按缺失值数量升序排序
    elif X_col == [] and vars_to_fill: #仅指定需插补的列
        columns_to_drop = ['LE', 'NEE', 'H', 'SHF', 'u*']
        mete_data = dataset.drop(columns=[col for col in columns_to_drop if col in dataset.columns], axis=1)
        data_to_fill = dataset.loc[:, vars_to_fill]
        missing_values = data_to_fill.isnull().sum() # 计算需插补的列每列的缺失值数量
        sorted_vars_with_missing = missing_values[missing_values > 0].sort_values(ascending=True).index.tolist() # 筛选出含缺失值的列并按缺失值数量升序排序
    else: #有缺失值的列全插补
        columns_to_drop = ['LE', 'NEE', 'H', 'SHF', 'u*']
        mete_data = dataset.drop(columns=[col for col in columns_to_drop if col in dataset.columns], axis=1)
        missing_values = mete_data.isnull().sum() # 计算每列的缺失值数量
        sorted_vars_with_missing = missing_values[missing_values > 0].sort_values(ascending=True).index.tolist() # 筛选出含缺失值的列并按缺失值数量升序排序
    
    if not sorted_vars_with_missing:
        print("No variables with missing values to fill.")
        return dataset

    #插补(从缺失值最少的开始)
    for var_to_fill in sorted_vars_with_missing:
        in_mete_data = mete_data.copy(deep=True) #创建 mete_data 的深拷贝以避免修改原数据
        in_mete_data.loc[:, in_mete_data.columns != var_to_fill] = in_mete_data.loc[:, in_mete_data.columns != var_to_fill].fillna(num_to_fill_na) #将除var_to_fill以外列的空值赋值
        
        if gap_filling_method in ['XGboost', 'XGB', 'XGb']:
            gap_filling_method = 'XGB'
            filled_mete_data = XGboostGapFilling(in_mete_data, var_to_fill, Mode, X_col, pred_all=False)
        elif gap_filling_method in ['RandomForest', 'RF']:
            gap_filling_method = 'RF'
            filled_mete_data = RandomForestGapFilling(in_mete_data, var_to_fill, Mode, X_col, pred_all=False)
        elif gap_filling_method in ['Adaboost', 'Ada']:
            gap_filling_method = 'Ada'
            filled_mete_data = AdaboostGapFilling(in_mete_data, var_to_fill, Mode, X_col, pred_all=False)
        elif gap_filling_method in ['NeuralNetwork', 'ANN']:
            gap_filling_method = 'ANN'
            filled_mete_data = ANNGapFilling(in_mete_data, var_to_fill, Mode, X_col, pred_all=False)

        mete_data[var_to_fill] = mete_data[var_to_fill].combine_first(filled_mete_data[var_to_fill + '_' + gap_filling_method + 'GapFilled'])
        print(f"{var_to_fill} Gap Filling Finished")

    mete_data_gapfilled = mete_data.loc[:, sorted_vars_with_missing]
    mete_data_gapfilled = mete_data_gapfilled.add_suffix('_' + gap_filling_method + 'GapFilled')
    mete_data_gapfilled['datetime'] = mete_data.loc[:, 'datetime']
    dataset_GapFilled = pd.merge(dataset, mete_data_gapfilled, on='datetime', how='outer')

    dataset_GapFilled['datetime'] = pd.to_datetime(dataset_GapFilled['datetime'])  # 确保 datetime 列是时间格式
    dataset_GapFilled = dataset_GapFilled.sort_values(by='datetime')  # 按时间排序

    return dataset_GapFilled


if __name__ == "__main__":
    ERA5file = r'E:\term1\flux_postprocessing\proccess_demo_DBS\DBS_ERA5_202201010800_202407010700_full.csv'
    rawfile = r'E:\term1\flux_postprocessing\proccess_demo_DBS\DBS_L2_24_1.csv'
    dataset = BuildDataset(ERA5file, rawfile)
    dataset.to_csv(r"E:\term1\flux_postprocessing\proccess_demo_DBS\dataset.csv", index=False, mode="w")

    dataset_GapFill_NEE = XGboostGapFilling(dataset, 'NEE')
    dataset_GapFilled = GapFillCols(dataset, gap_filling_method='XGB', Mode='use_era5_only', X_col=[])
    dataset_GapFilled.to_csv(r"E:\term1\flux_postprocessing\proccess_demo_DBS\dataset_GapFilled.csv", index=False, mode="w")