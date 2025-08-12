import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('seaborn-v0_8')

def MPT(data, ustar_col='u*', nee_col='NEE', tair_col='Tair', rg_col='Rg',  datetime_col='datetime', 
        start_time=None, end_time=None, temp_groups=6, nee_bins=20, vegetation_type='forest', visualization=True):
    """
    MPT(移动点)U*阈值判定方法

    参数：
        data: pd.DataFrame, 包含所需列的数据
        ustar_col: str, 摩擦风速列的名称
        nee_col: str, NEE列的名称
        rg_col: str, 总辐射列的名称
        datetime_col: str, 时间列的名称
        start_time: str, 可选, 筛选的起始时间, 格式为 'yyyy-mm-dd hh:mm:ss'
        end_time: str, 可选, 筛选的结束时间, 格式为 'yyyy-mm-dd hh:mm:ss'
        temp_groups: int, 温度分组数量(等样本大小), 默认6
        bins: int, 将 u* 分箱的数量, 默认20
        vegetation_type: str, 植被类型, 默认为forest
        visualization: bool, 是否可视化, 默认True

    返回：
        ustar_threshold: float, 估算的 u* 阈值
    """

    # 转换时间列为 datetime 格式（如果尚未转换）
    data.loc[:,datetime_col] = pd.to_datetime(data.loc[:,datetime_col])
    # 根据时间区间筛选数据
    if start_time:
        start_time = pd.to_datetime(start_time)
        data = data[data[datetime_col] >= start_time]
    if end_time:
        end_time = pd.to_datetime(end_time)
        data = data[data[datetime_col] <= end_time]

    # 筛选夜间数据（净辐射 < 0）
    night_data = data[data[rg_col] < 10].copy()

    # 按温度分组（分位数法）
    night_data['Temp_group'] = pd.qcut(night_data[tair_col], q=temp_groups, labels=False)
    thresholds = []

    for group in range(temp_groups):
        # 提取当前温度组的数据
        temp_group_data = night_data[night_data['Temp_group'] == group]
        if temp_group_data.empty:
            thresholds.append(None)
            continue
        # 检查温度和 u* 的相关性
        correlation = temp_group_data[[tair_col, ustar_col]].corr().iloc[0, 1] #计算NEE于Ustar的相关系数
        if abs(correlation) >= 0.4: #若相关系数>=0.4就放弃这个温度区间
            print(f"Temperature group {group+1} rejected due to strong correlation (|r| = {abs(correlation):.2f} >= 0.4).")
            thresholds.append(None)
            continue
        # 获取当前温度区间的最大值和最小值
        temp_min = temp_group_data[tair_col].min()
        temp_max = temp_group_data[tair_col].max()
    
        # 按 u* 分箱
        temp_group_data = temp_group_data.copy()  # 显式创建副本
        temp_group_data.loc[:, 'Ustar_bin'] = pd.qcut(temp_group_data[ustar_col], q=nee_bins)
        # temp_group_data['Ustar_bin'] = pd.qcut(temp_group_data[ustar_col], q=nee_bins)
        
        # 计算分箱统计量
        bin_stats = temp_group_data.groupby('Ustar_bin', observed=False).agg(
            mean_nee=(nee_col, 'mean'),
            count=(ustar_col, 'count')
        ).reset_index()
        bin_stats['Ustar_mid'] = bin_stats['Ustar_bin'].apply(lambda x: (x.left + x.right) / 2)
        # print(bin_stats)

        # 初始化 u* 阈值
        ustar_threshold = None

        # 判断条件逐步放宽
        for tolerance in [0.99, 0.90]:
            for i in range(len(bin_stats)):
                # 当前分箱的 NEE 平均值
                current_nee_mean = bin_stats['mean_nee'].iloc[i]
                # 更高u*分箱的 NEE 平均值
                higher_bins_nee_mean = bin_stats['mean_nee'].iloc[i+1:].mean()

                # 判断当前分箱是否超过更高分箱 NEE 的 {tolerance * 100}%
                if current_nee_mean > tolerance * higher_bins_nee_mean:
                    ustar_threshold = bin_stats['Ustar_mid'].iloc[i]
                    break
            
            # 如果找到阈值则退出循环
            if ustar_threshold is not None:
                break

        # 如果仍未找到阈值, 则采用最低建议值, 得到的ustar_threshold为该温度组的ustar_threshold
        if ustar_threshold is None:
            ustar_threshold = 0.1 if vegetation_type == 'forest' else 0.01
            print(f'The threshold for this temperature group was not found, use the minimum recommended value {ustar_threshold}.')
        thresholds.append(ustar_threshold)

        # 可视化结果
        if visualization:
            plt.figure(figsize=(10, 6))
            plt.scatter(temp_group_data[ustar_col], temp_group_data[nee_col], marker='o', label='NEE', color='gray', alpha=0.4, s=15)
            plt.scatter(bin_stats['Ustar_mid'], bin_stats['mean_nee'], marker='+', label='NEE (Mean)', color='blue')
            plt.axhline(higher_bins_nee_mean, color='green', linestyle='--', label='Higher bins NEE Mean', alpha=0.4)
            plt.axvline(ustar_threshold, color='red', linestyle='-', label=f'U* Threshold = {ustar_threshold:.3f}')
            plt.xlabel('u* (m/s)')
            plt.ylabel('NEE (umol CO2 m^-2 s^-1)')
            plt.title(f'MPT U* Threshold for Temperature Group {group+1} (Temp Range: {temp_min:.2f} to {temp_max:.2f}°C)')
            plt.legend()
            # plt.grid()
            plt.show()

    # 最终阈值为所有温度组的中位数
    thresholds = [t for t in thresholds if t is not None]  # 去除 None 值
    final_threshold = np.median(thresholds) if thresholds else None
    if visualization:
        print('Final MPT U* Threshold: ' + str(final_threshold))

    return final_threshold

def Bootstrap_ustar_threshold(data, ustar_col='u*', nee_col='NEE', tair_col='Tair', rg_col='Rn',  datetime_col='datetime', season=None, 
                               temp_groups=6, nee_bins=20, n_bootstraps=100, vegetation_type='forest', random_seed=1):
    '''
    通过 Bootstrap 重采样法计算 u* 阈值的置信区间，并进行季节性分析。

    函数描述：
        使用 Bootstrap 方法对数据进行重采样，计算每个季节的最大 u* 阈值。
        通过 MPT 方法分析数据，并计算每个季节和全年的 u* 阈值的5%、50%（中位数）和95%置信区间。
        可视化不同季节的 u* 阈值与 NEE 的关系，并在图中标出 u5、u50、u95。

    参数：
        data (pd.DataFrame): 包含 u*、NEE、Tair、Rn、datetime 等数据的 DataFrame。
        ustar_col (str): u* 列的列名，默认为 'u*'。
        nee_col (str): NEE 列的列名，默认为 'NEE'。
        tair_col (str): Tair 列的列名，默认为 'Tair'。
        rg_col (str): Rn 列的列名，默认为 'Rn'。
        datetime_col (str): 日期时间列的列名，默认为 'datetime'。
        season (list of tuples): 季节的起止日期，如果为 None 或空列表，则默认使用全年数据。
            每个季节为一个 (start_time, end_time) 元组。
        temp_groups (int): 温度分组的数量，默认为 6。
        nee_bins (int): NEE 数据的分组数量，默认为 20。
        n_bootstraps (int): Bootstrap 重采样次数，默认为 100。
        vegetation_type (str): 植被类型，默认为 'forest'。影响 MPT 分析。
        random_seed (int or None): 随机种子，默认为 1。如果为 None，则使用随机种子。

    返回：
        tuple, 包含以下内容：
        bootstrap_thresholds: 每次重采样计算得到的最大 u* 阈值列表。
        u5: 全年数据的 5% 分位数。
        u50: 全年数据的中位数（50% 分位数）。
        u95: 全年数据的 95% 分位数。
        seasonal_results: 季节性分析结果字典，每个季节包含 u5、u50 和 u95 的值。
    '''

    if random_seed is not None:
        np.random.seed(random_seed)

    # 计算一年的总数据点数（假设半小时间隔）
    total_points = len(data)  # data 的行数即为数据条数, 通常为一年 17520 个半小时点（24 * 365 * 2）
    # 初始化存储 Bootstrap 阈值的列表
    bootstrap_thresholds = []

    # 如果没有提供季节，则默认处理全部数据
    data.loc[:,datetime_col] = pd.to_datetime(data.loc[:,datetime_col])
    if season is None or season == []:
        season = [(data[datetime_col].min(), data[datetime_col].max())]
        
    # 初始化季节性结果字典
    seasonal_thresholds_all = {f"{start_time} to {end_time}": [] for start_time, end_time in season}

    # 开始 Bootstrap 重采样
    for i in range(n_bootstraps):
        # 进行Bootstrap重采样
        bootstrap_sample = data.sample(n=total_points, replace=True) #从data中抽total_points给样本，有放回
        # 季节性分析：每次重采样会分别针对每个季节计算 u* 阈值
        seasonal_thresholds = []

        for start_time, end_time in season:
            # 筛选当前季节数据
            seasonal_data = bootstrap_sample[(bootstrap_sample[datetime_col] >= pd.to_datetime(start_time)) & 
                                            (bootstrap_sample[datetime_col] <= pd.to_datetime(end_time))]

            # 调用 MPT 函数计算当前季节的 U* 阈值
            ustar_threshold = MPT(seasonal_data, ustar_col=ustar_col, nee_col=nee_col, tair_col=tair_col, rg_col=rg_col, 
                                  datetime_col=datetime_col, vegetation_type=vegetation_type, start_time=start_time, 
                                  end_time=end_time, temp_groups=temp_groups, nee_bins=nee_bins, visualization=False)
            
            # 将计算得到的阈值保存
            if ustar_threshold is not None:
                seasonal_thresholds.append(ustar_threshold) # 保存用于计算总体阈值
                seasonal_thresholds_all[f"{start_time} to {end_time}"].append(ustar_threshold) # 保存用于计算各季节阈值

        # 获取当前Bootstrap样本中各季节的最大 U* 阈值
        if seasonal_thresholds:
            max_seasonal_threshold = max(seasonal_thresholds)
            bootstrap_thresholds.append(max_seasonal_threshold) 
        
    # 计算每个季节的5%, 95%置信区间和中位数 (u5, u50, u95)
    if season is not None:
        seasonal_results = {}
        for season_range, thresholds in seasonal_thresholds_all.items():
            if thresholds:
                u5 = np.percentile(thresholds, 5)   # 5% 分位数
                u50 = np.median(thresholds)          # 中位数
                u95 = np.percentile(thresholds, 95)  # 95% 分位数
                seasonal_results[season_range] = {'u5': u5, 'u50': u50, 'u95': u95}
            else:
                seasonal_results[season_range] = {'u5': None, 'u50': None, 'u95': None}

    # 计算总体5%, 95%置信区间和中位数
    u5 = np.percentile(bootstrap_thresholds, 5)   # 5% 分位数
    u95 = np.percentile(bootstrap_thresholds, 95)  # 95% 分位数
    u50 = np.median(bootstrap_thresholds)  # 中位数

    # 打印结果并画图
    if season is not None:
        #画图
        print(f'len season {len(season)}')
        season_num = len(season)
        cols = 1 # 每行的子图数量(列数)
        rows = season_num  # 行数
        plt.figure(figsize=(cols * 8, rows * 3))    # 动态调整图大小
        idx = 1
        for season_range, result in seasonal_results.items():
            # 分季节画图
            start_time, end_time = season_range.split(" to ")
            # 获取当前季节的数据
            seasonal_data = data[(data[datetime_col] >= pd.to_datetime(start_time)) & (data[datetime_col] <= pd.to_datetime(end_time))]
            # 创建散点图
            plt.subplot(rows, cols, idx)
            plt.scatter(seasonal_data[ustar_col], seasonal_data[nee_col], marker='o', label='NEE', color='gray', alpha=0.4, s=15)
            # 绘制u5, u50, u95
            if result['u5'] is not None:
                plt.axvline(result['u5'], color='purple', linestyle='-', label=f'U* 5% = {result["u5"]:.3f}')
            if result['u50'] is not None:
                plt.axvline(result['u50'], color='red', linestyle='-', label=f'U* 50% = {result["u50"]:.3f}')
            if result['u95'] is not None:
                plt.axvline(result['u95'], color='blue', linestyle='-', label=f'U* 95% = {result["u95"]:.3f}')
            # 设置图例和标签
            plt.xlabel('u* (m/s)')
            plt.ylabel('NEE (umol CO2 m^-2 s^-1)')
            plt.title(f'u* threshold for {season_range}')
            plt.legend()
            idx = idx + 1
        plt.tight_layout()
        plt.show()
        # 分季节打印结果
        for season_range, result in seasonal_results.items():
            print(f"Season: {season_range} u5: {result['u5']:.3f}, u50: {result['u50']:.3f}, u95: {result['u95']:.3f}")
    # 打印全年结果
    print(f"Total: u5: {u5:.3f}, u50: {u50:.3f}, u95: {u95:.3f}")

    return bootstrap_thresholds, u5, u95, u50, seasonal_results

if __name__ == "__main__":
    dataset = pd.read_csv(r"E:\term1\flux_postprocessing\proccess_demo_DBS\dataset_23.csv")
    ustar_threshold = MPT(dataset, ustar_col='u*', nee_col='NEE', tair_col='Tair', rg_col='Rn',  datetime_col='datetime', 
                          start_time=None, end_time=None, temp_groups=6, nee_bins=20, vegetation_type='forest', visualization=True)
    bootstrap_thresholds, u5, u95, u50 = Bootstrap_ustar_threshold(dataset, ustar_col='u*', nee_col='NEE', tair_col='Tair', rg_col='Rn',  datetime_col='datetime', season=None, 
                               temp_groups=6, nee_bins=20, n_bootstraps=100, vegetation_type='forest', random_seed=1)