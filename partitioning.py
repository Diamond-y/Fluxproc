import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

def lloyd_taylor(T, Rref, E0, Tref=10, T0=-46.02):
    return Rref * np.exp(E0 * (1 / (Tref - T0) - 1 / (T - T0)))

def estimate_Rref(df, T_col, NEE_col, Rg_col, E0, window_days=7, shift_days=4):
    """
    使用指定的E0和固定时间窗口估算夜间数据中的参考温度下的呼吸速率Rref。
    """
    Rref_estimates = []
    dates = []

    Rref_window_size = window_days*48
    Rref_shift = shift_days*48
    for start_idx in range(0, len(df) - Rref_window_size, Rref_shift):
        end_idx = start_idx + Rref_window_size
        window_df = df.iloc[start_idx:end_idx]

        # # 如果窗口小于全窗口，使用右半部分的数据
        # if len(window_df) < Rref_window_size:
        #     right_half_window_df = df.iloc[-Rref_window_size:]  # 使用右半部分数据
        #     window_df = right_half_window_df

        # 判断该窗口内的夜间数据（Rg < 10）
        night_data = window_df[window_df[Rg_col] < 10]

        # 如果窗口内数据过少，则跳过此窗口
        if len(night_data) < 3:
            continue

        try:
            # 使用Lloyd-Taylor模型拟合Rref，仅用夜间数据
            popt, _ = curve_fit(
                lambda T, Rref: lloyd_taylor(T, Rref, E0),
                night_data[T_col],
                night_data[NEE_col],
                bounds=(0, 1e3)  # 限定Rref的范围在0到1000之间
            )
            Rref_estimates.append(popt[0])
            # 记录窗口的中心点处的时间，后面赋值到该时间
            dates.append(window_df.index[int(len(window_df) / 2)])  # 中心时间点
        except Exception as e:
            print(f"拟合Rref时出错: {e}")
            continue

    if not Rref_estimates:
        print("无法估算Rref。")
        return None  # 如果Rref无法估算，返回None

    # 创建包含日期索引的Series
    rref_series = pd.Series(Rref_estimates, index=dates)

    ### 将 rref_series 的索引调整为与 df 的索引一致，并进行线性插值
    full_index = df.index  # 使用 df 的时间索引作为插值的基准
    rref_series = rref_series.reindex(full_index)
    # rref_series_filled = rref_series.reindex(full_index).interpolate(method='time')
    # 使用线性插值方法填充所有的 NaN 值，包括最开始的部分
    rref_series_filled = rref_series.interpolate(method='linear', limit_direction='both')

    # ### 将 rref_series 的索引调整为与 df 的索引一致，并进行样条函数插值
    # full_index = df.index  # 使用 df 的时间索引作为插值的基准
    # rref_series = rref_series.reindex(full_index)
    # # 获取已知的时间点和对应的值（去除NaN值）
    # valid_idx = rref_series.dropna().index
    # valid_values = rref_series.dropna().values
    # # 生成样条插值函数
    # spline = InterpolatedUnivariateSpline(
    #     (valid_idx - valid_idx[0]).total_seconds(),  # 转换为秒，以便插值
    #     valid_values,
    #     k=2  # 二次样条插值（默认是二次样条）
    # )
    # # 使用样条函数插值填充整个时间序列
    # rref_series_filled_values = spline((full_index - valid_idx[0]).total_seconds())
    # # 生成与df时间索引一致的Series
    # rref_series_filled = pd.Series(rref_series_filled_values, index=full_index)

    return rref_series_filled


# 主要函数
def NT_Reichstein(data, datetime_col, NEE_col, T_col, Rg_col, night_estimate=True, min_temp_range=5, E0_window_days=15, E0_shift_days=5, Rref_window_days=7, Rref_shift_days=4):
    """
    基于 Reichstein 等人(2005)的方法，将 NEE 分解为 GPP 和 RECO。

    参数：
        data (pd.DataFrame): 输入的数据框(DataFrame)
        datetime_col (str): datetime列的列名，该列格式形如'yyyy-mm-dd hh:mm:ss'
        NEE_col (str): NEE(净生态系统交换)列的列名
        T_col (str): 温度列的列名
        Rg_col (str): 短波辐射列的列名
        night_estimate (bool): 如果为False，夜间不计算RECO和GPP，直接用NEE列做RECO，夜间GPP为0(默认为True)
        min_temp_range (int): 温度范围的最小值(默认为5°C)
        E0_window_days (int): 估算E0的窗口大小（单位为天）
        E0_shift_days (int): E0窗口的滑动步长（单位为天）
        Rref_window_days (int): 估算Rref的窗口大小（单位为天）
        Rref_shift_days (int): Rref窗口的滑动步长（单位为天）

    返回：
        包含RECO和GPP列的DataFrame
    """
    df = data.copy(deep=True)  # 创建 data 的深拷贝以避免修改原数据

    # 将datetime列转换为pandas的datetime格式并设置为索引
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.set_index(datetime_col, inplace=True)

    # 清理数据，将无效数据转换为NaN
    df = df.replace([None], np.nan)

    # 初始化RECO和GPP列，设置为NaN
    df['RECO'] = np.nan
    df['GPP'] = np.nan

    # 估算E0(温度敏感性参数)
    E0_estimates = []
    E0_valid_values = []  # 存储有效的E0值及其标准误差

    E0_window_size = E0_window_days * 48  # 每个窗口的数据量
    E0_shift = E0_shift_days * 48  # 每次滑动的步长

    for start_idx in range(0, len(df) - E0_window_size, E0_shift):
        end_idx = start_idx + E0_window_size
        window_df = df.iloc[start_idx:end_idx]

        # 选择夜间数据（Rg < 10）
        night_data = window_df[window_df[Rg_col] < 10]

        # 确保窗口内数据足够且温度变化大于 min_temp_range
        if len(night_data) < 6 or night_data[T_col].max() - night_data[T_col].min() < min_temp_range:
            continue

        try:
            # 使用Lloyd-Taylor模型拟合E0，Rref设为1，之后会处理
            popt, pcov = curve_fit(
                lambda T, E0: lloyd_taylor(T, Rref=1, E0=E0),
                night_data[T_col], night_data[NEE_col],
                bounds=(30, 450)  # E0的范围在30到450之间
            )

            # 获取拟合的E0值和标准误差（标准误差是协方差矩阵的对角元素的平方根）
            E0_val = popt[0]
            E0_std_err = np.sqrt(np.diag(pcov))[0]  # 获取E0的标准误差

            # 计算相对标准误差
            E0_rel_error = E0_std_err / E0_val if E0_val != 0 else np.nan

            # 判断E0是否在有效范围内且相对标准误差小于50%
            if 30 <= E0_val <= 450 and E0_rel_error < 0.5:
                E0_estimates.append(E0_val)
                E0_valid_values.append((E0_val, E0_rel_error))
        except Exception as e:
            print(f"拟合E0时出错: {e}")
            continue

    # 如果没有有效的E0估算，返回原始数据
    if not E0_estimates:
        print("无法估算E0。")
        return df

    # 根据标准误差排序，选择前三个最小标准误差的E0估算值
    E0_valid_values.sort(key=lambda x: x[1])  # 按标准误差排序
    best_E0_values = [e[0] for e in E0_valid_values[:3]]  # 取标准误差最小的前三个E0

    # 计算这些E0值的平均值作为全局E0
    E0 = np.mean(best_E0_values)
    print(f'global E0: {E0}')

    # 估算Rref(参考温度下的呼吸)，使用7天窗口，每4天连续滑动一次进行估算
    Rref_series = estimate_Rref(df, T_col, NEE_col, Rg_col, E0, window_days=Rref_window_days, shift_days=Rref_shift_days)

    if Rref_series is None:
        print("无法继续处理，因为无法估算Rref。")
        return df

    # 将 Rref_series 的索引调整为与 df 的索引一致并插值
    # Rref_series = Rref_series.reindex(df.index).interpolate(method='time')

    # 遍历数据框，计算RECO和GPP
    for idx, row in df.iterrows():
        Tair = row[T_col]

        try:
            Rref = Rref_series.loc[idx]
        except KeyError:
            print(f"索引 {idx} 不在 Rref_series 中，将 Rref 设置为 NaN")
            Rref = np.nan

        if row[Rg_col] < 10 and not night_estimate:  # 如果是夜间并且night_estimate为False
            RECO = row[NEE_col]  # 夜间直接使用NEE列做RECO
            GPP = 0  # 夜间GPP为0
        else:
            if pd.isna(Rref):
                RECO = np.nan
                GPP = np.nan
            else:        
                RECO = lloyd_taylor(Tair, Rref, E0) # 使用Lloyd-Taylor模型估算RECO
                GPP = RECO - row[NEE_col]  # 估算GPP

        df.at[idx, 'RECO'] = RECO
        df.at[idx, 'GPP'] = GPP

    # 恢复 datetime 列并重置索引
    df.reset_index(inplace=True)
    
    return df, Rref_series


def light_response_curve(rg, alpha, beta, gamma):
    """矩形双曲光响应曲线。"""
    return (alpha * beta * rg) / (alpha * rg + beta) + gamma

def adjust_beta(vpd, beta0, k, vpd0=10):
    """根据 VPD 调整 beta。"""
    return beta0 * np.exp(-k * (vpd - vpd0)) if vpd > vpd0 else beta0

def DT_Lasslop(data, datetime_col, nee_col, tair_col, rg_col, vpd_col, window_size=15, step_size=2):
    """
    实现 Lasslop et al. (2010) 方法，将 NEE 分解为 GPP 和 RECO。
    
    参数：
        data (pd.DataFrame): 输入数据，包括 NEE、Tair、Rg 和 VPD 列。
        datetime_col: datetime列的列名，该列格式形如'yyyy-mm-dd hh:mm:ss'
        nee_col (str): NEE（净生态系统交换）列名。
        tair_col (str): 气温（°C）列名。
        rg_col (str): 短波辐射（W/m²）列名。
        vpd_col (str): 饱和水汽压差（hPa）列名。
        window_size (int): 移动窗口大小（天）。
        step_size (int): 移动窗口步长（天）。

    返回：
        pd.DataFrame: 包含新增 GPP 和 RECO 列的数据框。
        list: 更新后的e0_estimations
    """
    # 预处理数据：解析时间列并排序
    data = data.copy()
    data[datetime_col] = pd.to_datetime(data[datetime_col])
    data = data.sort_values(datetime_col).reset_index(drop=True)
    
    # 初始化输出列
    data['GPP'] = np.nan
    data['RECO'] = np.nan
    e0_estimations = []
    
    # 定义Lloyd-Taylor温度响应函数
    def lloyd_taylor(temp, R_ref, E0):
        T_ref = 10.0 + 273.15  # 参考温度转换为K
        T0 = 227.13            # 经验常数
        return R_ref * np.exp(E0 * (1/(T_ref - T0) - 1/(temp + 273.15 - T0)))  # temp单位需为°C
    
    # 定义直角双曲线光响应函数
    def light_response(PAR, alpha, GPP_max, beta=0):
        return (alpha * PAR * GPP_max) / (alpha * PAR + GPP_max) - beta
    
    # 生成移动窗口的时间范围
    start_time = data[datetime_col].min()
    end_time = data[datetime_col].max()
    window_centers = pd.date_range(start=start_time, end=end_time, freq=f'{step_size}D')
    
    # 存储窗口参数
    params_log = []
    
    # 遍历每个窗口
    for center in window_centers:
        # 计算窗口边界
        window_start = center - pd.Timedelta(days=window_size//2)
        window_end = center + pd.Timedelta(days=window_size//2)
        window_data = data[(data[datetime_col] >= window_start) & 
                          (data[datetime_col] <= window_end)]
        
        if len(window_data) < 10:  # 忽略数据不足的窗口
            continue
        
        # --- 步骤1：夜间数据拟合RECO参数 ---
        night_data = window_data[window_data[rg_col] < 10]  # 假设Rg<10为夜间
        if len(night_data) > 5:
            try:
                # 非线性拟合
                popt, _ = curve_fit(lloyd_taylor, 
                                   night_data[tair_col], 
                                   night_data[nee_col],
                                   p0=[5, 300],   # 初始猜测：R_ref=5 μmol/m²/s, E0=300
                                   maxfev=1000)
                R_ref, E0 = popt
                e0_estimations.append(E0)
            except:
                R_ref, E0 = np.nan, np.nan
        else:
            R_ref, E0 = np.nan, np.nan
        
        # --- 步骤2：白天数据拟合GPP参数 ---
        day_data = window_data[window_data[rg_col] >= 10]
        if not day_data.empty and not np.isnan(R_ref):
            try:
                # 计算白天的RECO（使用夜间拟合参数）
                T_day = day_data[tair_col].values
                RECO_day = lloyd_taylor(T_day, R_ref, E0)
                
                # 定义目标函数：预测NEE = RECO - GPP
                def predict_nee(params, PAR, VPD):
                    alpha, GPP_max, beta = params
                    GPP = light_response(PAR, alpha, GPP_max, beta)
                    return RECO_day - GPP
                
                # 初始参数和边界
                initial_guess = [0.1, 30, 0.05]  # alpha, GPP_max, beta
                bounds = ([0, 0, 0], [1, 100, 0.5])
                
                # 拟合光响应参数
                popt, _ = curve_fit(predict_nee, 
                                   (day_data[rg_col], day_data[vpd_col]), 
                                   day_data[nee_col],
                                   p0=initial_guess,
                                   bounds=bounds,
                                   maxfev=2000)
                
                # 记录窗口参数
                params_log.append({
                    'center': center,
                    'R_ref': R_ref,
                    'E0': E0,
                    'alpha': popt[0],
                    'GPP_max': popt[1],
                    'beta': popt[2]
                })
            except:
                params_log.append({'center': center, 'R_ref': R_ref, 'E0': E0, 
                                  'alpha': np.nan, 'GPP_max': np.nan, 'beta': np.nan})
        else:
            params_log.append({'center': center, 'R_ref': R_ref, 'E0': E0, 
                              'alpha': np.nan, 'GPP_max': np.nan, 'beta': np.nan})
    
    # --- 步骤3：参数插值 ---
    param_df = pd.DataFrame(params_log)
    param_df = param_df.dropna(subset=['R_ref', 'E0', 'alpha', 'GPP_max'])
    
    # 线性插值参数到原始时间序列
    if not param_df.empty:
        time_num = pd.to_numeric(data[datetime_col])
        for param in ['R_ref', 'E0', 'alpha', 'GPP_max', 'beta']:
            interp_func = interp1d(
                pd.to_numeric(param_df['center']), 
                param_df[param], 
                kind='linear', 
                fill_value="extrapolate"
            )
            data[param] = interp_func(time_num)
        
        # --- 步骤4：计算GPP和RECO ---
        # 计算全时间序列的RECO
        data['RECO'] = lloyd_taylor(data[tair_col], data['R_ref'], data['E0'])
        
        # 计算全时间序列的GPP
        day_mask = data[rg_col] >= 10
        data['GPP'] = np.where(
            day_mask,
            light_response(data[rg_col], data['alpha'], data['GPP_max'], data['beta']),
            np.nan
        )
        
        # 根据NEE=RECO-GPP调整夜间RECO
        data.loc[~day_mask, 'RECO'] = data[nee_col]
        data.loc[day_mask, 'GPP'] = data.loc[day_mask, 'RECO'] - data.loc[day_mask, nee_col]
    
    return data, e0_estimations