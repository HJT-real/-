"""
任务一：总体经济发展趋势展示
生成6个静态图表，展示中国2015-2024年GDP变化趋势
（已移除图表1、4、5的面积填充效果，图表1、4纵坐标从0开始）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 全局配置
# ============================================================
# 中文字体设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 统一配色
COLOR_PRIMARY = '#1f77b4'      # 蓝色 - 主色/第二产业
COLOR_SECONDARY = '#ff7f0e'    # 橙色 - 第三产业
COLOR_TERTIARY = '#2ca02c'     # 绿色 - 第一产业
COLOR_ACCENT = '#d62728'       # 红色 - 强调色
COLOR_BG = '#fafafa'           # 背景色

# 产业配色（保持一致）
INDUSTRY_COLORS = {
    '第一产业': '#2ca02c',
    '第二产业': '#1f77b4',
    '第三产业': '#ff7f0e',
}

# 输出目录
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), '使用数据')

# ============================================================
# 数据读取
# ============================================================
def load_data():
    """读取所有数据文件"""
    # 全国GDP及产业占比统计表
    df_gdp = pd.read_csv(
        os.path.join(DATA_DIR, '2015-2024年全国GDP及产业占比统计表.csv'),
        encoding='utf-8-sig'
    )
    df_gdp.columns = df_gdp.columns.str.strip()

    # 国内生产总值构成占比
    df_structure = pd.read_csv(
        os.path.join(DATA_DIR, '2015-2024年国内生产总值构成占比.csv'),
        encoding='utf-8-sig'
    )
    df_structure.columns = df_structure.columns.str.strip()

    # 全国GDP细分行业数据
    df_detail = pd.read_csv(
        os.path.join(DATA_DIR, '全国GDP细分行业数据（2015-2024）.csv'),
        encoding='utf-8-sig'
    )
    df_detail.columns = df_detail.columns.str.strip()

    return df_gdp, df_structure, df_detail


# ============================================================
# 图表1：GDP总量随时间发展图（仅折线图）
# ============================================================
def chart1_gdp_trend(df_gdp):
    """GDP总量随时间的发展图"""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    years = df_gdp['年份'].values
    gdp = df_gdp['全国 GDP_亿元'].values / 10000  # 转换为万亿元

    # 折线 (已去除面积填充)
    ax.plot(years, gdp, color=COLOR_PRIMARY, linewidth=2.8, marker='o',
            markersize=8, markerfacecolor='white', markeredgewidth=2.2,
            markeredgecolor=COLOR_PRIMARY, zorder=5)

    # 关键节点标注
    key_points = {2015: gdp[0], 2020: gdp[5], 2024: gdp[-1]}
    for yr, val in key_points.items():
        ax.annotate(f'{val:.1f} 万亿元',
                    xy=(yr, val), xytext=(0, 18),
                    textcoords='offset points', ha='center',
                    fontsize=11, fontweight='bold', color=COLOR_PRIMARY,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=COLOR_PRIMARY, alpha=0.9))

    # 2020年拐点标注
    ax.annotate('疫情影响\n增速放缓',
                xy=(2020, gdp[5]), xytext=(40, -40),
                textcoords='offset points',
                fontsize=9, color=COLOR_ACCENT,
                arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff0f0',
                          edgecolor=COLOR_ACCENT, alpha=0.8))

    ax.set_xlabel('年份', fontsize=13, fontweight='bold')
    ax.set_ylabel('GDP总量（万亿元）', fontsize=13, fontweight='bold')
    ax.set_title('2015-2024年中国GDP总量变化趋势', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(years)
    ax.set_xlim(2014.5, 2024.5)
    # 纵坐标从0开始
    ax.set_ylim(0, max(gdp) * 1.15)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 添加增长幅度标注
    growth = (gdp[-1] / gdp[0] - 1) * 100
    ax.text(0.98, 0.05, f'十年累计增长 {growth:.1f}%',
            transform=ax.transAxes, fontsize=12, ha='right',
            color=COLOR_PRIMARY, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor=COLOR_PRIMARY, alpha=0.9))

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, '图表1_GDP总量趋势.png')
    plt.savefig(filepath)
    plt.close()
    print(f'[OK] 图表1已保存: {filepath}')


# ============================================================
# 图表2：GDP产业结构随时间发展图（堆叠面积图）
# ============================================================
def chart2_industry_structure(df_gdp):
    """GDP产业结构随时间发展图"""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    years = df_gdp['年份'].values
    p1 = df_gdp['第一产业占比 %'].values
    p2 = df_gdp['第二产业占比 %'].values
    p3 = df_gdp['第三产业占比 %'].values

    # 堆叠面积图
    ax.stackplot(years, p1, p2, p3,
                 labels=['第一产业', '第二产业', '第三产业'],
                 colors=[INDUSTRY_COLORS['第一产业'],
                         INDUSTRY_COLORS['第二产业'],
                         INDUSTRY_COLORS['第三产业']],
                 alpha=0.75)

    # 在每个年份的堆叠区域内标注具体占比数值
    for i, year in enumerate(years):
        # 动态调整左右两端的对齐方式，防止文字溢出面积图变为不可见
        if i == 0:  # 2015年左对齐并微调位置
            ha_align = 'left'
            x_offset = 0.05
        elif i == len(years) - 1:  # 2024年右对齐并微调位置
            ha_align = 'right'
            x_offset = -0.05
        else:  # 中间年份居中对齐
            ha_align = 'center'
            x_offset = 0

        x_pos = year + x_offset

        # 第一产业
        ax.text(x_pos, p1[i] / 2, f'{p1[i]:.1f}%',
                ha=ha_align, va='center', fontsize=9,
                color='white', fontweight='bold')
        # 第二产业
        ax.text(x_pos, p1[i] + p2[i] / 2, f'{p2[i]:.1f}%',
                ha=ha_align, va='center', fontsize=9,
                color='white', fontweight='bold')
        # 第三产业
        ax.text(x_pos, p1[i] + p2[i] + p3[i] / 2, f'{p3[i]:.1f}%',
                ha=ha_align, va='center', fontsize=10,
                color='white', fontweight='bold')

    ax.set_xlabel('年份', fontsize=13, fontweight='bold')
    ax.set_ylabel('产业占比（%）', fontsize=13, fontweight='bold')
    ax.set_title('2015-2024年中国GDP产业结构演变', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(years)
    ax.set_xlim(2014.5, 2025.5)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 图例
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9,
              edgecolor='gray', fancybox=True)

    # 趋势箭头标注
    ax.annotate('第三产业占比持续上升 ↑',
                xy=(2019, 78), fontsize=12, color=COLOR_SECONDARY,
                fontweight='bold', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='none', alpha=0.8))
    ax.annotate('第二产业占比持续下降 ↓',
                xy=(2019, 35), fontsize=12, color=COLOR_PRIMARY,
                fontweight='bold', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='none', alpha=0.8))

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, '图表2_GDP产业结构演变.png')
    plt.savefig(filepath)
    plt.close()
    print(f'[OK] 图表2已保存: {filepath}')


# ============================================================
# 图表3：GDP年增长率变化柱状图
# ============================================================
def chart3_growth_rate(df_gdp):
    """GDP年增长率变化图"""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    years = df_gdp['年份'].values
    gdp = df_gdp['全国 GDP_亿元'].values

    # 计算同比增长率
    growth_years = years[1:]
    growth_rates = np.diff(gdp) / gdp[:-1] * 100

    # 颜色：2020年用红色，其他用蓝色渐变
    colors = []
    for i, yr in enumerate(growth_years):
        if yr == 2020:
            colors.append(COLOR_ACCENT)
        else:
            # 渐变蓝色
            intensity = 0.4 + 0.6 * (i / len(growth_years))
            colors.append(COLOR_PRIMARY)

    bars = ax.bar(growth_years, growth_rates, width=0.6, color=colors,
                  edgecolor='white', linewidth=1.2, zorder=3)

    # 柱顶标注
    for bar, rate in zip(bars, growth_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f'{rate:.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                color=COLOR_ACCENT if rate < 5 else COLOR_PRIMARY)

    # 均值参考线
    mean_rate = np.mean(growth_rates)
    ax.axhline(y=mean_rate, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.text(2024.5, mean_rate + 0.15, f'均值 {mean_rate:.1f}%',
            fontsize=9, color='gray', ha='right', fontstyle='italic')

    # 2020年特殊标注
    idx_2020 = list(growth_years).index(2020)
    ax.annotate('新冠疫情冲击\n增速骤降至 {0:.1f}%'.format(growth_rates[idx_2020]),
                xy=(2020, growth_rates[idx_2020]),
                xytext=(30, -50), textcoords='offset points',
                fontsize=10, color=COLOR_ACCENT, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff0f0',
                          edgecolor=COLOR_ACCENT, alpha=0.9))

    ax.set_xlabel('年份', fontsize=13, fontweight='bold')
    ax.set_ylabel('GDP同比增长率（%）', fontsize=13, fontweight='bold')
    ax.set_title('2016-2024年中国GDP同比增长率变化', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(growth_years)
    ax.set_ylim(0, max(growth_rates) * 1.25)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, '图表3_GDP年增长率.png')
    plt.savefig(filepath)
    plt.close()
    print(f'[OK] 图表3已保存: {filepath}')


# ============================================================
# 图表4：人均GDP变化趋势折线图 (已去除面积填充)
# ============================================================
def chart4_per_capita_gdp(df_detail):
    """人均GDP变化趋势图"""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    years = df_detail['年份'].values
    per_capita = df_detail['人均 GDP_元'].values / 10000  # 转换为万元

    # 折线 (已去除面积填充)
    ax.plot(years, per_capita, color='#9467bd', linewidth=2.8, marker='s',
            markersize=8, markerfacecolor='white', markeredgewidth=2.2,
            markeredgecolor='#9467bd', zorder=5)

    # 起止标注
    ax.annotate(f'{per_capita[0]:.2f} 万元\n(约 {df_detail["人均 GDP_元"].values[0]/6900:.1f} 千美元)',
                xy=(years[0], per_capita[0]), xytext=(30, -30),
                textcoords='offset points', fontsize=10, fontweight='bold',
                color='#9467bd',
                arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#9467bd', alpha=0.9))

    ax.annotate(f'{per_capita[-1]:.2f} 万元\n(约 {df_detail["人均 GDP_元"].values[-1]/7100:.1f} 千美元)',
                xy=(years[-1], per_capita[-1]), xytext=(-80, 20),
                textcoords='offset points', fontsize=10, fontweight='bold',
                color='#9467bd',
                arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#9467bd', alpha=0.9))

    # 1万美元参考线（约71000元 ≈ 7.1万元）
    usd_10000 = 71000 / 10000
    ax.axhline(y=usd_10000, color=COLOR_ACCENT, linestyle=':', linewidth=1.5, alpha=0.6)
    ax.text(2014.6, usd_10000 + 0.1, '1万美元参考线',
            fontsize=9, color=COLOR_ACCENT, fontstyle='italic')

    # 找到跨越1万美元的年份
    cross_idx = np.where(per_capita >= usd_10000)[0]
    if len(cross_idx) > 0:
        cross_year = years[cross_idx[0]]
        cross_val = per_capita[cross_idx[0]]
        ax.plot(cross_year, cross_val, marker='*', markersize=15,
                color=COLOR_ACCENT, zorder=6)
        ax.annotate(f'{int(cross_year)}年突破1万美元',
                    xy=(cross_year, cross_val), xytext=(30, 25),
                    textcoords='offset points', fontsize=10,
                    color=COLOR_ACCENT, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=1.2),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff0f0',
                              edgecolor=COLOR_ACCENT, alpha=0.8))

    ax.set_xlabel('年份', fontsize=13, fontweight='bold')
    ax.set_ylabel('人均GDP（万元）', fontsize=13, fontweight='bold')
    ax.set_title('2015-2024年中国人均GDP变化趋势', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(years)
    ax.set_xlim(2014.5, 2024.5)
    # 纵坐标从0开始
    ax.set_ylim(0, max(per_capita) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 增长幅度
    growth = (per_capita[-1] / per_capita[0] - 1) * 100
    ax.text(0.02, 0.95, f'十年增长 {growth:.1f}%',
            transform=ax.transAxes, fontsize=12, color='#9467bd', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#9467bd', alpha=0.9))

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, '图表4_人均GDP趋势.png')
    plt.savefig(filepath)
    plt.close()
    print(f'[OK] 图表4已保存: {filepath}')


# ============================================================
# 图表5：GDP累计增长倍数图（以2015年为基期）(已去除面积填充)
# ============================================================
def chart5_growth_multiple(df_gdp):
    """以2015年为基期的GDP累计增长倍数图"""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    years = df_gdp['年份'].values
    gdp = df_gdp['全国 GDP_亿元'].values
    p1 = df_gdp['第一产业_亿元'].values
    p2 = df_gdp['第二产业_亿元'].values
    p3 = df_gdp['第三产业_亿元'].values

    # 以2015年为基期计算增长倍数
    base_gdp = gdp[0]
    base_p1 = p1[0]
    base_p2 = p2[0]
    base_p3 = p3[0]

    mult_gdp = gdp / base_gdp
    mult_p1 = p1 / base_p1
    mult_p2 = p2 / base_p2
    mult_p3 = p3 / base_p3

    # 绘制各产业增长倍数
    ax.plot(years, mult_p1, color=INDUSTRY_COLORS['第一产业'], linewidth=2.2,
            marker='o', markersize=6, markerfacecolor='white',
            markeredgewidth=1.8, label='第一产业', zorder=4)
    ax.plot(years, mult_p2, color=INDUSTRY_COLORS['第二产业'], linewidth=2.2,
            marker='s', markersize=6, markerfacecolor='white',
            markeredgewidth=1.8, label='第二产业', zorder=4)
    ax.plot(years, mult_p3, color=INDUSTRY_COLORS['第三产业'], linewidth=2.2,
            marker='^', markersize=7, markerfacecolor='white',
            markeredgewidth=1.8, label='第三产业', zorder=4)

    # GDP总量增长倍数（加粗）
    ax.plot(years, mult_gdp, color='#333333', linewidth=3.2,
            marker='D', markersize=8, markerfacecolor='white',
            markeredgewidth=2.2, label='GDP总量', zorder=5)

    # 基期参考线
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(2014.6, 1.02, '基期 (2015=1.0)', fontsize=9, color='gray')

    # 终值标注
    for label, mult, color, offset in [
        ('GDP总量', mult_gdp[-1], '#333333', (15, 10)),
        ('第一产业', mult_p1[-1], INDUSTRY_COLORS['第一产业'], (15, -15)),
        ('第二产业', mult_p2[-1], INDUSTRY_COLORS['第二产业'], (15, -15)),
        ('第三产业', mult_p3[-1], INDUSTRY_COLORS['第三产业'], (15, 10)),
    ]:
        ax.annotate(f'{label} {mult:.2f}倍',
                    xy=(2024, mult), xytext=offset,
                    textcoords='offset points', fontsize=9,
                    fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=color, alpha=0.85))

    ax.set_xlabel('年份', fontsize=13, fontweight='bold')
    ax.set_ylabel('增长倍数（2015年=1.0）', fontsize=13, fontweight='bold')
    ax.set_title('2015-2024年GDP及三大产业累计增长倍数\n（以2015年为基期）',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(years)
    ax.set_xlim(2014.5, 2025.5)
    # 纵坐标保持原样（下限0.9）
    ax.set_ylim(0.9, max(mult_p3) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9,
              edgecolor='gray', fancybox=True)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, '图表5_GDP累计增长倍数.png')
    plt.savefig(filepath)
    plt.close()
    print(f'[OK] 图表5已保存: {filepath}')


# ============================================================
# 图表6：三大产业对GDP增长贡献率图
# ============================================================
def chart6_industry_contribution(df_gdp):
    """三大产业对GDP增长的贡献率图"""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    years = df_gdp['年份'].values
    gdp = df_gdp['全国 GDP_亿元'].values
    p1 = df_gdp['第一产业_亿元'].values
    p2 = df_gdp['第二产业_亿元'].values
    p3 = df_gdp['第三产业_亿元'].values

    # 计算各产业对GDP增量的贡献率
    contrib_years = years[1:]
    delta_gdp = np.diff(gdp)
    delta_p1 = np.diff(p1)
    delta_p2 = np.diff(p2)
    delta_p3 = np.diff(p3)

    contrib_p1 = delta_p1 / delta_gdp * 100
    contrib_p2 = delta_p2 / delta_gdp * 100
    contrib_p3 = delta_p3 / delta_gdp * 100

    # 堆叠柱状图
    bar_width = 0.55
    bars1 = ax.bar(contrib_years, contrib_p1, bar_width,
                   label='第一产业', color=INDUSTRY_COLORS['第一产业'],
                   edgecolor='white', linewidth=0.8, zorder=3)
    bars2 = ax.bar(contrib_years, contrib_p2, bar_width,
                   bottom=contrib_p1, label='第二产业',
                   color=INDUSTRY_COLORS['第二产业'],
                   edgecolor='white', linewidth=0.8, zorder=3)
    bars3 = ax.bar(contrib_years, contrib_p3, bar_width,
                   bottom=contrib_p1 + contrib_p2, label='第三产业',
                   color=INDUSTRY_COLORS['第三产业'],
                   edgecolor='white', linewidth=0.8, zorder=3)

    # 在每根柱子内部标注百分比（仅标注>8%的）
    for i, yr in enumerate(contrib_years):
        # 第一产业
        if contrib_p1[i] > 5:
            ax.text(yr, contrib_p1[i] / 2, f'{contrib_p1[i]:.1f}%',
                    ha='center', va='center', fontsize=8, color='white',
                    fontweight='bold')
        # 第二产业
        if contrib_p2[i] > 5:
            ax.text(yr, contrib_p1[i] + contrib_p2[i] / 2, f'{contrib_p2[i]:.1f}%',
                    ha='center', va='center', fontsize=8, color='white',
                    fontweight='bold')
        # 第三产业
        ax.text(yr, contrib_p1[i] + contrib_p2[i] + contrib_p3[i] / 2,
                f'{contrib_p3[i]:.1f}%', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')

    # 50%参考线
    ax.axhline(y=50, color='gray', linestyle=':', linewidth=1.2, alpha=0.5)
    ax.text(2024.5, 51, '50%', fontsize=9, color='gray', fontstyle='italic')

    ax.set_xlabel('年份', fontsize=13, fontweight='bold')
    ax.set_ylabel('对GDP增长贡献率（%）', fontsize=13, fontweight='bold')
    ax.set_title('2016-2024年三大产业对GDP增长贡献率', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(contrib_years)
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9,
              edgecolor='gray', fancybox=True)

    # 添加解读文字
    avg_p3_contrib = np.mean(contrib_p3)
    ax.text(0.98, 0.95,
            f'第三产业平均贡献率: {avg_p3_contrib:.1f}%\n是经济增长的主要驱动力',
            transform=ax.transAxes, fontsize=10, ha='right', va='top',
            color=COLOR_SECONDARY, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor=COLOR_SECONDARY, alpha=0.9))

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, '图表6_产业增长贡献率.png')
    plt.savefig(filepath)
    plt.close()
    print(f'[OK] 图表6已保存: {filepath}')


# ============================================================
# 主程序
# ============================================================
def main():
    print('=' * 60)
    print('任务一：总体经济发展趋势展示 - 图表生成')
    print('=' * 60)

    # 读取数据
    df_gdp, df_structure, df_detail = load_data()
    print(f'\n[INFO] 数据加载完成:')
    print(f'  - 全国GDP及产业占比: {len(df_gdp)} 条记录')
    print(f'  - GDP构成占比: {len(df_structure)} 条记录')
    print(f'  - GDP细分行业: {len(df_detail)} 条记录')
    print()

    # 生成图表
    chart1_gdp_trend(df_gdp)
    chart2_industry_structure(df_gdp)
    chart3_growth_rate(df_gdp)
    chart4_per_capita_gdp(df_detail)
    chart5_growth_multiple(df_gdp)
    chart6_industry_contribution(df_gdp)

    print('\n' + '=' * 60)
    print('[DONE] 任务一全部6个图表生成完毕！')
    print('=' * 60)


if __name__ == '__main__':
    main()