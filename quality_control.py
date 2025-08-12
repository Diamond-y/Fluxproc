import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt; plt.style.use('seaborn-v0_8')

def smartY_qc(dataset, col_to_qc):
    """
    对任意DataFrame的每一列计算Smart Y-range, 并进行qc
    Smart Y-range = 中位数 ± 10 * IQR, 超出范围的值设为NaN
    
    参数：
        dataset (pd.DataFrame): 输入的DataFrame
        col_to_qc (list): 需要进行qc的列

    返回：
        pd.DataFrame: qc后的DataFrame
    """
    qc_method = '_SmartY_qc'

    dataset = dataset.copy()  # 创建副本, 避免修改原数据
    dataset_qc = dataset.loc[:, col_to_qc]

    for column in dataset_qc.columns:
        if dataset_qc[column].dtype in ['float64', 'int64']:  # 仅处理数值列
            median = dataset_qc[column].median()
            iqr = dataset_qc[column].quantile(0.75) - dataset_qc[column].quantile(0.25)
            lower_bound = median - 10 * iqr
            upper_bound = median + 10 * iqr
            qced_column = column + qc_method
            dataset[qced_column] = dataset_qc[column].apply(lambda x: x if lower_bound <= x <= upper_bound else None)
    
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])  # 转换为时间类型
    dataset = dataset.sort_values(by='datetime')  # 时间顺序排序

    #绘制qc结果图
    plot_qc_results(dataset, col_to_qc, qc_method)
    return dataset

def MaxMinValue_qc(dataset, col_to_qc, lower_bound=None, upper_bound=None):
    """
    对任意DataFrame的一些列手动输入Max、Min值, 并进行qc
    超出范围的值设为NaN
    
    参数：
        dataset (pd.DataFrame): 输入的DataFrame
        col_to_qc (list): 需要进行qc的列
        lower_bound (list): 下限 (可选), 与col_to_qc对应
        upper_bound (list): 上限 (可选), 与col_to_qc对应

    返回：
        pd.DataFrame: qc后的DataFrame
    """
    qc_method = '_MaxMinValue_qc'

    dataset['datetime'] = pd.to_datetime(dataset['datetime'])  # 转换为时间类型
    dataset = dataset.sort_values(by='datetime')  # 时间顺序排序

    if lower_bound is None:
        lower_bound = [None] * len(col_to_qc)
    if upper_bound is None:
        upper_bound = [None] * len(col_to_qc)

    dataset_qc = dataset.loc[:, col_to_qc].copy() # 创建副本, 避免修改原数据
    for column_num, column in enumerate(dataset_qc.columns):
        if dataset_qc[column].dtype in ['float64', 'int64']:  # 仅处理数值列
            if lower_bound[column_num] is None or upper_bound[column_num] is None:
                #计算用于参考的统计值
                median = dataset_qc[column].median()
                upper_quantile = dataset_qc[column].quantile(0.75)
                lower_quantile = dataset_qc[column].quantile(0.25)
                #画数据的图用于参考
                plt.figure(figsize=(20,8))
                plt.scatter(dataset['datetime'], dataset[column], label=column, color='blue', alpha=0.6, s=15)
                plt.axhline(y=upper_quantile, color='green', linestyle='--', linewidth=1.5, label=f'upper_quantile:{upper_quantile:.2f}', alpha=0.6,)
                plt.axhline(y=median, color='brown', linestyle='--', linewidth=1.5, label=f'median:{median:.2f}', alpha=0.6,)
                plt.axhline(y=lower_quantile, color='orange', linestyle='--', linewidth=1.5, label=f'lower_quantile:{lower_quantile:.2f}', alpha=0.6,)
                plt.title(f"Variable being proccessed: {column}", fontsize=20)
                # 确定目标 x 轴标签数量
                xtick_num = 45  # 目标显示的标签数量
                step = max(1, len(dataset['datetime']) // xtick_num)  # 根据目标数量计算步长
                # 设置 x 轴标签
                plt.xticks(dataset['datetime'][::step], rotation=45)
                plt.ylabel(column, fontsize=16)
                yticks = np.linspace(np.nanmin(dataset[column]), np.nanmax(dataset[column]), 20)  # 生成20个刻度
                plt.yticks(yticks, fontsize=16)
                plt.legend(fontsize=12)
                plt.show()

                print(f'Variable being proccessed: {column}')
                lower_bound[column_num] = float(input('lower_bound: '))
                upper_bound[column_num] = float(input('upper_bound: '))
                
            qced_column = column + qc_method
            dataset[qced_column] = dataset_qc[column].apply(lambda x: x if lower_bound[column_num] <= x <= upper_bound[column_num] else None)

    #绘制qc结果图
    plot_qc_results(dataset, col_to_qc, qc_method)
    return dataset

def Dependency_qc(dataset, col_to_qc, dependency_col, lower_bound=None, upper_bound=None):
    """
    根据依赖关系进行质量控制
    DataFrame[dependency_col]中每一列列需手动输入Max、Min值, 根据datetime列的日期, 超出范围部分的col_to_qc列的元素值会被设为NaN
    
    参数：
        dataset (pd.DataFrame): 输入的DataFrame
        col_to_qc (string): 需要进行qc的列(本函数一次仅能处理一列!!!)
        dependency_col (list): col_to_qc依赖的列
        lower_bound (list): 下限 (可选), 与dependency_col对应
        upper_bound (list): 上限 (可选), 与dependency_col对应

    返回：
        pd.DataFrame: qc后的DataFrame
    """
    qc_method = '_Dependency_qc'

    dataset['datetime'] = pd.to_datetime(dataset['datetime'])  # 转换为时间类型
    dataset = dataset.sort_values(by='datetime')  # 时间顺序排序

    dataset = dataset.copy()  # 创建副本, 避免修改原数据

    if lower_bound is None:
        lower_bound = [None] * len(dependency_col)
    if upper_bound is None:
        upper_bound = [None] * len(dependency_col)

    # 遍历 dependency_col 的每一列
    for column in dependency_col:
        column_num = 0
        if dataset[column].dtype in ['float64', 'int64']:  # 仅处理数值列
            # 画被qc列和参考列的数据图, 用于手动确定上下界
            plt.figure(figsize=(15,12))

            # 被qc列
            # 计算被qc列的统计值
            median_qc = dataset[col_to_qc].median()
            upper_quantile_qc = dataset[col_to_qc].quantile(0.75)
            lower_quantile_qc = dataset[col_to_qc].quantile(0.25)
            plt.subplot(211)
            plt.scatter(dataset['datetime'], dataset[col_to_qc], label=col_to_qc, color='blue', alpha=0.6, s=15)
            plt.axhline(y=upper_quantile_qc, color='green', linestyle='--', linewidth=1.5, label=f'upper_quantile:{upper_quantile_qc:.2f}', alpha=0.6,)
            plt.axhline(y=median_qc, color='brown', linestyle='--', linewidth=1.5, label=f'median:{median_qc:.2f}', alpha=0.6,)
            plt.axhline(y=lower_quantile_qc, color='orange', linestyle='--', linewidth=1.5, label=f'lower_quantile:{lower_quantile_qc:.2f}', alpha=0.6,)
            plt.title(f"col_to_qc: {col_to_qc}", fontsize=20)
            # 确定目标 x 轴标签数量
            xtick_num = 45  # 目标显示的标签数量
            step = max(1, len(dataset['datetime']) // xtick_num)  # 根据目标数量计算步长
            # 设置 x 轴标签
            plt.xticks(dataset['datetime'][::step], rotation=45)
            plt.ylabel(col_to_qc, fontsize=16)
            yticks = np.linspace(np.nanmin(dataset[col_to_qc]), np.nanmax(dataset[col_to_qc]), 15)  # 生成15个刻度
            plt.yticks(yticks, fontsize=16)
            plt.legend(fontsize=14)

            # 参考列
            # 计算参考列的统计值
            median = dataset[column].median()
            upper_quantile = dataset[column].quantile(0.75)
            lower_quantile = dataset[column].quantile(0.25)
            plt.subplot(212)
            plt.scatter(dataset['datetime'], dataset[column], label=column, color='blue', alpha=0.6, s=15)
            plt.axhline(y=upper_quantile, color='green', linestyle='--', linewidth=1.5, label=f'upper_quantile:{upper_quantile:.2f}', alpha=0.6,)
            plt.axhline(y=median, color='brown', linestyle='--', linewidth=1.5, label=f'median:{median:.2f}', alpha=0.6,)
            plt.axhline(y=lower_quantile, color='orange', linestyle='--', linewidth=1.5, label=f'lower_quantile:{lower_quantile:.2f}', alpha=0.6,)
            plt.title(f"Dependency: {column}", fontsize=20)
            # 确定目标 x 轴标签数量
            xtick_num = 45  # 目标显示的标签数量
            step = max(1, len(dataset['datetime']) // xtick_num)  # 根据目标数量计算步长
            # 设置 x 轴标签
            plt.xticks(dataset['datetime'][::step], rotation=45)
            plt.ylabel(column, fontsize=16)
            yticks = np.linspace(np.nanmin(dataset[column]), np.nanmax(dataset[column]), 15)  # 生成15个刻度
            plt.yticks(yticks, fontsize=16)
            plt.legend(fontsize=14)

            plt.tight_layout()
            plt.show()

            # 手动输入上下界
            if lower_bound is None or lower_bound[column_num] is None:
                lower_bound[column_num] = float(input(f'Enter lower_bound for dependency_col({column}): '))
            if upper_bound is None or upper_bound[column_num] is None:
                upper_bound[column_num] = float(input(f'Enter upper_bound for dependency_col({column}): '))

            # 标记符合条件(不用被删)的日期
            valid_dates = dataset[(dataset[column] >= lower_bound[column_num]) & (dataset[column] <= upper_bound[column_num])]['datetime']

            # 对 col_to_qc 列进行过滤, 非符合日期范围的值设为 NaN
            dataset[f"{col_to_qc}{qc_method}"] = dataset.apply(
                lambda row: row[col_to_qc] if row['datetime'] in valid_dates.values else None, axis=1
            )
        column_num = column_num + 1

    # 绘制结果
    plot_qc_results(dataset, [col_to_qc], qc_method, withbound=False)
    return dataset

def Datetime_qc(dataset, col_to_qc, date_ranges):
    """
    根据日期范围进行质量控制
    对 col_to_qc 列, 根据用户输入的日期范围, 清洗超出范围的数据, 将其设为 NaN
    
    参数：
        dataset (pd.DataFrame): 输入的 DataFrame。
        col_to_qc (string): 需要进行qc的列(本函数一次仅能处理一列!!!)
        date_ranges (list): 形如 [(开始时间1, 结束时间1),(开始时间2, 结束时间2)] 的列表, 用于筛选时间范围
    
    返回：
        pd.DataFrame: qc后的 DataFrame
    """
    qc_method = '_Datetime_qc'
    
    # 确保 datetime 列为时间类型并排序
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    dataset = dataset.sort_values(by='datetime').copy()
    
    # 验证日期范围格式
    valid_ranges = []
    # 合并所有合法的日期范围, 标记保留的数据
    invalid_mask = pd.Series(False, index=dataset.index)
    for start_date, end_date in date_ranges:
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            valid_ranges.append((start_date, end_date))
            invalid_mask |= (dataset['datetime'] >= start_date) & (dataset['datetime'] <= end_date)
        except Exception as e:
            print(f"Invalid date range ({start_date}, {end_date}): {e}")
    
    if not valid_ranges:
        raise ValueError("No valid date ranges provided.")
    
    # 清洗 col_to_qc 数据
    dataset[f"{col_to_qc}{qc_method}"] = dataset[col_to_qc].where(~invalid_mask, None)

    # 绘制清洗结果
    plot_qc_results(dataset, [col_to_qc], qc_method, withbound=False)
    
    return dataset


def despiking(data, col_to_qc, Rg_col, window_length, threshold_multiplier=3, process_period='both', replace_with_median=False):
    """
    对于一个pd.DataFrame中的指定列根据滑动窗口计算median_abs_deviation进行异常峰值检测并去峰值处理。
    
    参数:
        data : pd.DataFrame, 包含要处理的数据的 DataFrame。
        col_to_qc : str, 要进行去峰值处理的列名。
        Rg_col : str, 辐射列名，用于区分白天和夜晚。
        window_length : int。滑动窗口的长度。
        threshold_multiplier : float, optional, 定义异常值的 MAD 阈值倍数，默认为 3。
        process_period : str, optional, 如果为 'night'，则对夜晚的数据也进行去峰值处理；如果为'day'，则对白天的数据进行去峰值处理，如果为'both'，则对所有数据进行去峰值处理，默认为'both'.
        replace_with_median : bool, optional, 如果为 True，则用窗口内的中位数替换异常值；如果为 False，则将异常值置为 NaN，默认为 False.

    返回:
        pd.DataFrame, 处理后的 DataFrame，包含原始数据和去峰值处理后的数据。
    """
    qc_method = '_despiked'

    df = data.copy(deep=True)  # 创建 data 的深拷贝以避免修改原数据
    # 确保 datetime 列为时间类型并排序（如果您需要按时间排序的话）
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime').reset_index(drop=True)
    else:
        print("Warning: DataFrame does not contain a 'datetime' column.")

    def process(data_period):  # 注意传入的data_period仍是整数索引
        data_period['despiked'] = data_period[col_to_qc].copy()
        
        for i in range(0, len(data_period)):
            # 获取当前数据点的辐射值
            Rg_value = data_period.iloc[i][Rg_col]
            
            # 判断白天黑夜
            isday = True if Rg_value >= 10 else False
            
            if Rg_value < 10 and process_period == 'day':   # 如果是夜晚且不处理夜晚数据，则跳过该数据点
                continue
            elif Rg_value >= 10 and process_period == 'night':   # 如果是夜晚且不处理夜晚数据，则跳过该数据点
                continue
            elif process_period != 'day' and process_period != 'night' and process_period != 'both':
                raise ValueError("Invalid value for 'process_period'. It must be one of 'day', 'night', or 'both'. Please check the input.")
            
            # 计算当前窗口的起始和结束索引，考虑边缘情况
            start_idx = max(i - window_length // 2, 0)
            end_idx = min(i + window_length // 2 + 1, len(data_period))
            
            # 先切片，获取窗口内的数据
            window = data_period.iloc[start_idx:end_idx]
            # 若当前被qc的是白天的数据，只使用白天的数据判断其是否需被删，夜晚以此类推
            if isday:
                window_time = window[window[Rg_col] >= 10]
            else:
                window_time = window[window[Rg_col] < 10]
            # 选择数据列
            window_time_col = window_time[col_to_qc]
            # 忽略窗口中的 NaN 值
            window_clean = window_time_col.dropna()
            
            if len(window_clean) <= window_length // 8:  # 如果窗口内有效数据点过少，跳过
                continue
            
            mad = median_abs_deviation(window_clean, nan_policy='omit')
            median = window_clean.median()
            
            current_value = data_period.iloc[i][col_to_qc]
            if not np.isnan(current_value) and abs(current_value - median) > threshold_multiplier * mad:
                if replace_with_median:
                    # 用窗口内的中位数替代异常值
                    data_period.iat[i, data_period.columns.get_loc('despiked')] = median
                else:
                    # 将异常值置为 NaN
                    data_period.iat[i, data_period.columns.get_loc('despiked')] = np.nan
        
        return data_period

    # 对数据进行处理
    df = process(df)

    # 使用 merge 根据 datetime 列合并处理后的数据到原始 DataFrame 中
    df.rename(columns={'despiked': col_to_qc+qc_method}, inplace=True)
    df = df.sort_values(by='datetime').reset_index(drop=True)

    # 画图
    plot_qc_results(df, [col_to_qc], qc_method, withbound=False)
    
    return df


def plot_qc_results(dataframe, col_to_qc, qc_method, withbound=True):
    """
    绘制散点图展示数据qc前后的对比, 已删除和未删除的数据用不同颜色标记, 并标出上下界。
    """   
    dataframe['datetime'] = pd.to_datetime(dataframe['datetime'])  # 转换为时间类型
    dataframe = dataframe.sort_values(by='datetime')  # 时间顺序排序

    #画图
    num_qced_columns = len(col_to_qc)
    cols = 2 if withbound else 1 # 每行的子图数量
    rows = num_qced_columns  # 自动计算行数
    plt.figure(figsize=(cols * 15, rows * 4))    # 动态调整图大小
    for idx, col_name in enumerate(col_to_qc, start=1):
        # 原始数据和qc后的数据
        original_column = col_name
        qced_column = col_name + qc_method
        original_data = dataframe[original_column]
        qced_data = dataframe[qced_column]
        
        # 已删除的数据：qc后的数据为 NaN, 且原始数据不为 NaN
        removed_data = original_data[qced_data.isna()]
        kept_data = original_data[qced_data.notna()]

        #计算kept,removed,missing比例
        kept = (len(kept_data.dropna())/len(dataframe['datetime']))*100
        removed = (len(removed_data.dropna())/len(dataframe['datetime']))*100
        missing = (1-len(original_data.dropna())/len(dataframe['datetime']))*100

        if withbound:
            ## 左侧子图：已删除和未删除的数据
            plt.subplot(rows, cols, 2 * idx - 1)
            plt.scatter(dataframe['datetime'][kept_data.index], kept_data, color='blue', label='Kept Data', alpha=0.6, s=15)
            plt.scatter(dataframe['datetime'][removed_data.index], removed_data, color='red', label='Removed Data', alpha=0.6, s=15)

            # 添加上下界线
            lower_bound = dataframe[qced_column].min()
            upper_bound = dataframe[qced_column].max()
            plt.axhline(y=lower_bound, color='green', linestyle='--', linewidth=2.5, label=f'Lower Bound:{lower_bound:.3f}')
            plt.axhline(y=upper_bound, color='orange', linestyle='--', linewidth=2.5, label=f'Upper Bound:{upper_bound:.3f}')

            # 子图标题和标签
            plt.title(f"{original_column}{qc_method}_result | kept:{kept:.2f}% removed{removed:.2f}% missing:{missing:.2f}%", fontsize=14)
            # 确定目标 x 轴标签数量
            xtick_num = 45  # 目标显示的标签数量
            step = max(1, len(dataframe['datetime']) // xtick_num)  # 根据目标数量计算步长
            # 设置 x 轴标签
            plt.xticks(dataframe['datetime'][::step], rotation=45)
            plt.ylabel(original_column, fontsize=12)
            plt.legend(fontsize=12)

            ## 右侧子图：清洗后的数据
            plt.subplot(rows, cols, 2 * idx)
            plt.scatter(dataframe['datetime'][kept_data.index], kept_data, color='blue', label='Kept Data', alpha=0.6, s=15)

            # 子图标题和标签
            plt.title(f"{original_column}{qc_method}_result | kept:{kept:.2f}% removed{removed:.2f}% missing:{missing:.2f}%", fontsize=14)
            # 确定目标 x 轴标签数量
            xtick_num = 45  # 目标显示的标签数量
            step = max(1, len(dataframe['datetime']) // xtick_num)  # 根据目标数量计算步长
            # 设置 x 轴标签
            plt.xticks(dataframe['datetime'][::step], rotation=45)
            plt.ylabel(original_column, fontsize=12)
            plt.legend(fontsize=12)

        else:
            # 如果withbound=False, 每列仅生成一个子图展示所有数据
            plt.subplot(rows, cols, idx)
            plt.scatter(dataframe['datetime'][kept_data.index], kept_data, color='blue', label='Kept Data', alpha=0.6, s=15)
            plt.scatter(dataframe['datetime'][removed_data.index], removed_data, color='red', label='Removed Data', alpha=0.6, s=15)
        
            # 子图标题和标签
            plt.title(f"{original_column}{qc_method}_result | kept:{kept:.2f}% removed{removed:.2f}% missing:{missing:.2f}%", fontsize=14)
            # 确定目标 x 轴标签数量
            xtick_num = 45  # 目标显示的标签数量
            step = max(1, len(dataframe['datetime']) // xtick_num)  # 根据目标数量计算步长
            # 设置 x 轴标签
            plt.xticks(dataframe['datetime'][::step], rotation=45)
            plt.ylabel(original_column, fontsize=12)
            plt.legend(fontsize=12)

    # 调整布局
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset = pd.read_csv(r"E:\term1\flux_postprocessing\proccess_demo_DBS\dataset_23.csv")

    # smartY_qc
    col_to_qc = ['Tair', "Tsoil", "SWC", "PPFD", "RH", "VPD", "Rn", "Rg", "pr", "H", "LE", "NEE", "u*"]
    dataset_smartY_qc = smartY_qc(dataset, col_to_qc)
    # dataset_smartY_qc.to_csv(r"E:\term1\flux_postprocessing\proccess_demo_DBS\dataset_smartY_qc.csv", index=False, mode="w")

    # MaxMinValue_qc
    col_to_qc = ["NEE"]
    dataset_MaxMinValue_qc = MaxMinValue_qc(dataset, col_to_qc)

    # Dependency_qc
    col_to_qc='NEE'
    dependency_col = ["pr", "u*"]
    dataset_MaxMinValue_qc = Dependency_qc(dataset, col_to_qc=col_to_qc, dependency_col=dependency_col)

    # Datetime_qc
    col_to_qc = 'Tair'
    date_ranges = [("2024-01-20 00:00:00", "2024-02-10 00:00:00"), ("2024-04-01 08:30:00", "2024-04-20 10:00:00")]
    dataset_Datetime_qc = Datetime_qc(dataset, col_to_qc, date_ranges)
    dataset_Datetime_qc.to_csv(r"E:\term1\flux_postprocessing\proccess_demo_DBS\dataset_Datetime_qc.csv", index=False, mode="w")

    #despiking
    col_to_qc='NEE'
    dataset_despiking = despiking(dataset, col_to_qc=col_to_qc, Rg_col='Rg', window_length=48*10)