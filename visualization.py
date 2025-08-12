import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('default')
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

def plot_fingerprint(data, datetime_col, data_cols, cmap='viridis'):
    """
    绘制多个栅格图和右侧的折线图，显示每列数据的平均日分布。

    参数：
        data (pd.DataFrame): 包含 datetime 和数据列的数据框。
        datetime_col (str): 表示日期时间的列名。
        data_cols (list): 表示数据列名的列表。
        cmap (str): 颜色映射名称（默认 'viridis'）。
    """
    df = data.copy(deep=True) #创建 data 的深拷贝以避免修改原数据
    # 确保 datetime 列为 pandas 的 datetime 类型
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # 提取日期和时间
    df['date'] = df[datetime_col].dt.date
    df['time'] = df[datetime_col].dt.time

    # 创建主图和右侧折线图
    figsize=(8, 4)
    num_plots = len(data_cols)
    fig, axes = plt.subplots(
        num_plots, 2, figsize=(figsize[0] * 2, figsize[1] * num_plots),
        gridspec_kw={'width_ratios': [4, 1]}  # 设置热图和折线图的宽度比例
    )

    # 如果只有一个子图，axes 不是二维数组，需要调整
    if num_plots == 1:
        axes = [axes]

    # 遍历每一列数据
    for i, col in enumerate(data_cols):
        ax_heatmap = axes[i][0]  # 热图
        ax_lineplot = axes[i][1]  # 折线图

        # 构造热图数据
        pivot = df.pivot_table(index='time', columns='date', values=col, aggfunc='mean')
        X, Y = np.meshgrid(
            mdates.date2num(pivot.columns),  # 日期转为数字
            range(len(pivot.index))         # 时间的索引
        )
        Z = pivot.values

        # 绘制热图
        mesh = ax_heatmap.pcolormesh(X, Y, Z, shading='nearest', cmap=cmap)

        # 设置横纵坐标刻度
        x_ticks_num = 12
        y_ticks_num = 14
        # 横坐标
        x_tick_indices = np.linspace(0, len(pivot.columns) - 1, x_ticks_num, dtype=int)
        x_ticks = [mdates.date2num(pivot.columns[i]) for i in x_tick_indices]
        ax_heatmap.set_xticks(x_ticks)
        ax_heatmap.set_xticklabels([pivot.columns[i] for i in x_tick_indices])
        ax_heatmap.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # 纵坐标
        y_tick_indices = np.linspace(0, len(pivot.index) - 1, y_ticks_num, dtype=int)
        ax_heatmap.set_yticks(y_tick_indices)
        ax_heatmap.set_yticklabels([pivot.index[i] for i in y_tick_indices])

        # 添加热图标题和颜色条
        ax_heatmap.set_title(f'{col}', fontsize=18)
        cbar = fig.colorbar(mesh, ax=ax_heatmap, orientation='vertical', shrink=1) 
        cbar.set_label('')  # 去掉颜色条标签

        # 设置热图坐标轴标签
        ax_heatmap.set_xlabel('Date', fontsize=14)
        ax_heatmap.set_ylabel('Time', fontsize=14)

        ax_heatmap.grid(linestyle='--', alpha=0.5)

        # 构造折线图数据
        df['hour'] = df[datetime_col].dt.hour
        grouped = df.groupby('hour')[col]
        daily_mean = grouped.mean()
        daily_std = grouped.std()

        # 缩短误差棒（例如缩短为原误差值的 30%）
        scale_factor = 0.3
        adjusted_std = daily_std * scale_factor

        # 绘制折线图并添加误差棒（更短的 T 形）
        ax_lineplot.errorbar(
            daily_mean.index, daily_mean.values, yerr=adjusted_std.values,
            fmt='-o', label=f'Mean {col}', color='blue', ecolor='lightblue',
            capsize=2, elinewidth=1, capthick=0.5  # T 形样式
        )

        # 设置折线图标题和坐标轴
        ax_lineplot.set_title(f'{col} Daily Distribution', fontsize=16)
        ax_lineplot.set_xlabel('Hour', fontsize=14)
        ax_lineplot.set_ylabel('Mean Value', fontsize=14)

        # 设置横坐标仅显示 5 个刻度
        tick_positions = np.linspace(daily_mean.index.min(), daily_mean.index.max(), 5, dtype=int)
        ax_lineplot.set_xticks(tick_positions)
        ax_lineplot.set_xticklabels([f'{int(tick):02d}:00' for tick in tick_positions])
        ax_lineplot.grid(linestyle='--', alpha=0.5)

        # 添加图例到图内部
        # ax_lineplot.legend(loc='best')

        # 折线图只有最后一行子图显示横坐标刻度
        # if i != num_plots - 1:
        #     ax_lineplot.set_xticklabels([])

    # 调整整体布局
    plt.tight_layout()
    plt.show()


def plot_scatter(data, datetime_col, data_cols):
    """
    绘制多个栅格图和右侧的折线图，显示每列数据的平均日分布。

    参数：
        data (pd.DataFrame): 包含 datetime 和数据列的数据框。
        datetime_col (str): 表示日期时间的列名。
        data_cols (list): 表示数据列名的列表。
    """
    df = data.copy(deep=True) # 创建 data 的深拷贝以避免修改原数据
    # 确保 datetime 列为 pandas 的 datetime 类型
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    num_columns = len(data_cols)
    cols = 1 # 每行的子图数量
    rows = num_columns  # 自动计算行数
    plt.figure(figsize=(cols * 12, rows * 4))    # 动态调整图大小

    for i, col in enumerate(data_cols):
        # 绘制图形
        plt.subplot(rows, cols, i+1)
        plt.fill_between(df['datetime'], df[col], 0, where=(df[col]>=0), color='lightblue', alpha=0.5)
        plt.fill_between(df['datetime'], df[col], 0, where=(df[col]<=0), color='red', alpha=0.3)
        # plt.plot(df['DOY'], df[col], color='blue', linewidth=1, alpha=0.5)
        plt.scatter(df['datetime'], df[col], color='blue', alpha=0.2, s=15)

        # 确定目标 x 轴标签数量
        xtick_num = 30  # 目标显示的标签数量
        step = max(1, len(df['datetime']) // xtick_num)  # 根据目标数量计算步长
        # 设置 x 轴标签
        plt.xticks(df['datetime'][::step], rotation=45)

        # 添加图名、水平虚线和轴标签
        # plt.title(col)
        plt.axhline(0, color='gray', linestyle='-', linewidth=0.8)
        plt.ylabel(col, fontsize=14)
        plt.grid(linestyle='--', alpha=0.5)

    # 显示图形
    plt.tight_layout()
    plt.show()

def plot_carbon_fluxes(df, date_col='datetime', 
                       gpp_col='GPP', nee_col='NEE', reco_col='RECO',
                       output_name='carbon_fluxes', colors=None,
                       ylabel=r'flux (gC m$^{-2}$ d$^{-1}$)',
                       date_format='%Y%m%d'):
    """
    绘制碳通量时间序列图，兼容多种日期格式
    
    参数:
    df -- 包含时间序列数据的DataFrame
    date_col -- 日期时间列名 (默认'datetime')
    gpp_col -- GPP数据列名 (默认'GPP')
    nee_col -- NEE数据列名 (默认'NEE')
    reco_col -- RECO数据列名 (默认'RECO')
    output_name -- 输出文件名前缀 (默认'carbon_fluxes')
    colors -- 颜色字典 (默认使用预设配色)
    ylabel -- Y轴标签文本 (默认'flux (gC m$^{-2}$ d$^{-1}$)')
    date_format -- 日期格式字符串 (默认'%Y%m%d'，支持YYYYMMDD格式)
    """
    
    # 默认颜色
    if colors is None:
        colors = {
            'GPP': 'green',
            'RECO': 'darkred',
            'NEE': 'royalblue'
        }
    
    # 确保日期列是 datetime 类型
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            # 尝试使用指定的日期格式解析
            df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        except ValueError:
            # 如果指定格式失败，尝试自动解析
            df[date_col] = pd.to_datetime(df[date_col])
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(16, 6))

    # 绘制 GPP
    ax.plot(df[date_col], df[gpp_col], color=colors['GPP'], label='GPP', linewidth=1.5)
    ax.fill_between(df[date_col], df[gpp_col], alpha=0.2, color=colors['GPP'])

    # 绘制 RECO
    ax.plot(df[date_col], df[reco_col], color=colors['RECO'], label='RECO', linewidth=1.5)

    # 绘制 NEE
    ax.plot(df[date_col], df[nee_col], color=colors['NEE'], label='NEE', linewidth=1.5)
    ax.fill_between(df[date_col], df[nee_col], alpha=0.2, color=colors['NEE'])

    # y轴标签
    ax.set_ylabel(ylabel, fontsize=14)
    
    # 添加年份分隔虚线
    years = df[date_col].dt.year.unique()
    for year in years[1:]:
        ax.axvline(pd.Timestamp(f'{year}-01-01'), color='k', linestyle='--', linewidth=1)
    
    # 设置x轴格式 - 显示月份
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    ax.set_xlabel('Month', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')

    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # 保存图像为PNG和PDF格式
    plt.savefig(f'{output_name}.png', dpi=300)
    plt.savefig(f'{output_name}.pdf')
    
    plt.show()

def convert_halfhour_to_daily(df, date_col='datetime', 
                             gpp_col='GPP', nee_col='NEE', reco_col='RECO',
                             min_valid_hours=18):
    """
    将半小时尺度通量数据转换为天尺度，并转换单位为 gC m⁻² d⁻¹
    
    参数:
    df: 包含时间序列数据的DataFrame
    date_col: 日期时间列名 (默认'datetime')
    gpp_col: GPP数据列名 (默认'GPP')
    nee_col: NEE数据列名 (默认'NEE')
    reco_col: RECO数据列名 (默认'RECO')
    min_valid_hours: 每天所需的最小有效数据小时数 (默认18小时)
    
    返回:
    转换后的DataFrame，包含每日累积量
    """
    
    # 确保日期列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d %H:%M:%S')
    
    # 复制数据以避免修改原始DataFrame
    df = df.copy()
    
    # 单位转换系数 (μmol CO₂ 到 g CO₂)
    conversion_factor = 1800 * 12e-6
    
    # 应用单位转换（转换为每日累积量）
    for col in [gpp_col, nee_col, reco_col]:
        # 将瞬时通量转换为每日累积量
        df[col] = df[col] * conversion_factor
    
    # 设置日期索引
    df = df.set_index(date_col)
    
    # 计算每天的有效数据点数量 (半小时数据，一天最多48个点)
    min_valid_points = min_valid_hours * 2
    
    # 按天重采样并计算累积量
    daily_df = df.resample('D').agg({
        gpp_col: lambda x: x.sum() if x.count() >= min_valid_points else np.nan,
        nee_col: lambda x: x.sum() if x.count() >= min_valid_points else np.nan,
        reco_col: lambda x: x.sum() if x.count() >= min_valid_points else np.nan
    })
    
    # 添加日期列
    daily_df = daily_df.reset_index()
    daily_df.rename(columns={'index': date_col}, inplace=True)
    
    return daily_df

if __name__ == "__main__":
    # 示例数据
    data = pd.read_csv(r"E:\term1\flux_postprocessing\proccess_demo_DBS\dataset_23.csv")

    # 使用函数绘图
    plot_fingerprint(data, datetime_col='datetime', data_cols=['NEE', 'Tair'])
    plot_scatter(data, datetime_col='datetime', data_cols=['NEE', 'Tair'])