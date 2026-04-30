"""
任务三：相关分析
图表1：2024年各省/地区GDP与全国GDP局部到总体的构成关系（旭日图）
图表2：GDP变化与产业结构变化的相关性（散点矩阵图）
图表3：各省/地区GDP数据发展的相关性（相关矩阵热力图）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib.patches import Wedge, FancyArrowPatch
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 全局配置
# ============================================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

COLOR_BG = '#fafafa'
COLOR_PRIMARY = '#1f77b4'
COLOR_ACCENT = '#d62728'

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), '数据集')

FULL_TO_SHORT = {
    '北京市': '北京', '天津市': '天津', '河北省': '河北', '山西省': '山西',
    '内蒙古自治区': '内蒙古', '辽宁省': '辽宁', '吉林省': '吉林', '黑龙江省': '黑龙江',
    '上海市': '上海', '江苏省': '江苏', '浙江省': '浙江', '安徽省': '安徽',
    '福建省': '福建', '江西省': '江西', '山东省': '山东', '河南省': '河南',
    '湖北省': '湖北', '湖南省': '湖南', '广东省': '广东', '广西壮族自治区': '广西',
    '海南省': '海南', '重庆市': '重庆', '四川省': '四川', '贵州省': '贵州',
    '云南省': '云南', '西藏自治区': '西藏', '陕西省': '陕西', '甘肃省': '甘肃',
    '青海省': '青海', '宁夏回族自治区': '宁夏', '新疆维吾尔自治区': '新疆',
}

EAST_CHINA = ['北京', '天津', '河北', '上海', '江苏', '浙江', '福建', '山东', '广东', '海南']
CENTRAL_CHINA = ['山西', '安徽', '江西', '河南', '湖北', '湖南']
WEST_CHINA = ['内蒙古', '广西', '重庆', '四川', '贵州', '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆']
NORTHEAST = ['辽宁', '吉林', '黑龙江']

REGION_COLORS = {
    '东部': '#e74c3c',
    '中部': '#2ecc71',
    '西部': '#9b59b6',
    '东北': '#3498db',
}

def get_region(name):
    if name in EAST_CHINA:
        return '东部'
    elif name in CENTRAL_CHINA:
        return '中部'
    elif name in WEST_CHINA:
        return '西部'
    return '东北'


def load_data():
    df_province = pd.read_csv(
        os.path.join(DATA_DIR, '2015-2024年各省份GDP统计表.csv'),
        encoding='utf-8-sig'
    )
    df_province.columns = df_province.columns.str.strip()
    df_province['省份'] = df_province['省级行政区'].map(FULL_TO_SHORT)

    df_national = pd.read_csv(
        os.path.join(DATA_DIR, '2015-2024年全国GDP及产业占比统计表.csv'),
        encoding='utf-8-sig'
    )
    df_national.columns = df_national.columns.str.strip()

    return df_province, df_national


# ============================================================
# 图表1：旭日图——局部到总体的构成关系（2024年）
# ============================================================
def chart1_sunburst(df_province):
    """
    三层旭日图：
    内环 = 四大地区（东部、中部、西部、东北）
    中环 = 各省份（按地区归属）
    外环 = 各省份第一/二/三产业（颜色随地区变化，深浅区分产业）
    """
    df_2024 = df_province[df_province['年份'] == 2024].copy()
    df_2024['地区'] = df_2024['省份'].apply(get_region)
    df_2024 = df_2024.sort_values(['地区', '地区生产总值 (GDP)_亿元'], ascending=[True, False])

    total_gdp = df_2024['地区生产总值 (GDP)_亿元'].sum()

    # 地区分组
    region_order = ['东部', '中部', '西部', '东北']
    region_gdp = df_2024.groupby('地区')['地区生产总值 (GDP)_亿元'].sum()

    # 配色
    region_cmaps = {
        '东部': plt.cm.Reds,
        '中部': plt.cm.Greens,
        '西部': plt.cm.Purples,
        '东北': plt.cm.Blues,
    }

    fig, ax = plt.subplots(figsize=(16, 16))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.set_aspect('equal')
    ax.axis('off')

    # 半径设置
    r_inner = 0.25   # 内环（地区）内径
    r_mid1  = 0.45   # 内环外径 / 省份内径
    r_mid2  = 0.70   # 省份外径 / 产业内径
    r_outer = 0.90   # 产业外径

    start_angle = 90  # 从顶部开始，顺时针

    def draw_wedge(ax, theta1, theta2, r_in, r_out, color, alpha=1.0, lw=0.6):
        wedge = Wedge((0, 0), r_out, theta1, theta2,
                      width=r_out - r_in, facecolor=color, edgecolor='white',
                      linewidth=lw, alpha=alpha)
        ax.add_patch(wedge)

    def mid_angle(t1, t2):
        return (t1 + t2) / 2

    # ---- 内环：地区 ----
    cur = start_angle
    region_angles = {}
    for region in region_order:
        gdp = region_gdp.get(region, 0)
        span = gdp / total_gdp * 360
        t1 = cur - span
        t2 = cur
        color = REGION_COLORS[region]
        draw_wedge(ax, t1, t2, r_inner, r_mid1, color, alpha=0.9)
        # 内环标注
        ma = mid_angle(t1, t2)
        rx = (r_inner + r_mid1) / 2 * np.cos(np.radians(ma))
        ry = (r_inner + r_mid1) / 2 * np.sin(np.radians(ma))
        pct = gdp / total_gdp * 100
        ax.text(rx, ry, f'{region}\n{pct:.1f}%', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        region_angles[region] = (t1, t2)
        cur = t1

    # ---- 中环：省份 ----
    cur = start_angle
    province_angles = {}
    for region in region_order:
        df_reg = df_2024[df_2024['地区'] == region].sort_values('地区生产总值 (GDP)_亿元', ascending=False)
        reg_gdp = region_gdp.get(region, 1)
        cmap = region_cmaps[region]
        n = len(df_reg)
        for i, (_, row) in enumerate(df_reg.iterrows()):
            gdp = row['地区生产总值 (GDP)_亿元']
            span = gdp / total_gdp * 360
            t1 = cur - span
            t2 = cur
            color = cmap(0.4 + 0.5 * (n - 1 - i) / max(n - 1, 1))
            draw_wedge(ax, t1, t2, r_mid1, r_mid2, color, alpha=0.88)
            # 省份名标注（只标大省）
            if gdp / total_gdp > 0.015:
                ma = mid_angle(t1, t2)
                rx = (r_mid1 + r_mid2) / 2 * np.cos(np.radians(ma))
                ry = (r_mid1 + r_mid2) / 2 * np.sin(np.radians(ma))
                prov = row['省份']
                rotation = ma if -90 <= ma % 360 <= 90 else ma + 180
                ax.text(rx, ry, prov, ha='center', va='center',
                        fontsize=7, color='white', fontweight='bold',
                        rotation=rotation - 90)
            province_angles[row['省份']] = (t1, t2, region)
            cur = t1

    # ---- 外环：产业构成（随地区变化深浅） ----
    cur = start_angle
    for region in region_order:
        df_reg = df_2024[df_2024['地区'] == region].sort_values('地区生产总值 (GDP)_亿元', ascending=False)
        for _, row in df_reg.iterrows():
            gdp = row['地区生产总值 (GDP)_亿元']
            span = gdp / total_gdp * 360
            p1 = row['第一产业增加值_亿元'] / gdp if gdp > 0 else 0
            p2 = row['第二产业增加值_亿元'] / gdp if gdp > 0 else 0
            p3 = row['第三产业增加值_亿元'] / gdp if gdp > 0 else 0
            cmap = region_cmaps[region]

            # 三段：第一产业（浅）、第二（中）、第三（深）
            ind_colors = [cmap(0.35), cmap(0.6), cmap(0.85)]
            proportions = [p1, p2, p3]
            seg_cur = cur
            for prop, col in zip(proportions, ind_colors):
                seg_span = prop * span
                t1 = seg_cur - seg_span
                t2 = seg_cur
                if seg_span > 0.1:
                    draw_wedge(ax, t1, t2, r_mid2, r_outer, col, alpha=0.85, lw=0.4)
                seg_cur = t1
            cur = t1

    # 中心文字
    ax.text(0, 0.05, '全国', ha='center', va='center', fontsize=14,
            fontweight='bold', color='#333333')
    ax.text(0, -0.06, f'GDP\n{total_gdp/10000:.1f}万亿', ha='center', va='center',
            fontsize=10, color='#555555')

    # 图例：地区 + 各地区三种产业颜色（全部显示）
    legend_region = [mpatches.Patch(color=REGION_COLORS[r], label=r) for r in region_order]
    
    # 为每个地区生成三种产业颜色的图例
    legend_ind = []
    industry_labels = ['第一产业', '第二产业', '第三产业']
    industry_depths = [0.35, 0.6, 0.85]
    region_short = {'东部': '东', '中部': '中', '西部': '西', '东北': '东北'}
    for region in region_order:
        cmap = region_cmaps[region]
        short = region_short[region]
        for label, depth in zip(industry_labels, industry_depths):
            legend_ind.append(mpatches.Patch(
                color=cmap(depth), 
                label=f'{short}-{label}'
            ))

    all_handles = legend_region + legend_ind
    ax.legend(handles=all_handles, loc='upper center',
              bbox_to_anchor=(0.5, -0.04),
              ncol=8, fontsize=8.5,
              title='内环：地区          外环：产业深浅（按地区）',
              title_fontsize=9, framealpha=0.85,
              columnspacing=1.0, handlelength=1.3)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.15, 1.1)
    ax.set_title('2024年各省/地区GDP与全国GDP构成关系（旭日图）\n内环=地区，中环=省份，外环=三产业构成',
                 fontsize=13, fontweight='bold', color='#222222', y=0.98)

    out = os.path.join(OUTPUT_DIR, '图表1_GDP构成旭日图.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'已保存：{out}')


# ============================================================
# 图表2：GDP变化与产业结构变化的相关性（散点图矩阵）
# ============================================================
def chart2_correlation_scatter(df_province, df_national):
    """
    以各省份为观测单位，计算2015-2024年间：
    - GDP增速（%）
    - 第一/二/三产业占比变化（百分点）
    绘制4×4散点矩阵，展示相互关系。
    """
    df_2015 = df_province[df_province['年份'] == 2015].set_index('省份')
    df_2024 = df_province[df_province['年份'] == 2024].set_index('省份')

    provinces = df_2024.index.intersection(df_2015.index)

    df = pd.DataFrame(index=provinces)
    df['GDP增速(%)'] = ((df_2024['地区生产总值 (GDP)_亿元'] - df_2015['地区生产总值 (GDP)_亿元'])
                       / df_2015['地区生产总值 (GDP)_亿元'] * 100)
    df['第一产业占比变化(pp)'] = df_2024['第一产业占 GDP 比重_%'] - df_2015['第一产业占 GDP 比重_%']
    df['第二产业占比变化(pp)'] = df_2024['第二产业占 GDP 比重_%'] - df_2015['第二产业占 GDP 比重_%']
    df['第三产业占比变化(pp)'] = df_2024['第三产业占 GDP 比重_%'] - df_2015['第三产业占 GDP 比重_%']
    df['地区'] = [get_region(p) for p in provinces]

    variables = ['GDP增速(%)', '第一产业占比变化(pp)', '第二产业占比变化(pp)', '第三产业占比变化(pp)']
    n = len(variables)

    fig, axes = plt.subplots(n, n, figsize=(14, 14))
    fig.patch.set_facecolor(COLOR_BG)

    region_colors_map = {r: REGION_COLORS[r] for r in REGION_COLORS}
    colors = [region_colors_map[r] for r in df['地区']]

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            ax.set_facecolor(COLOR_BG)

            if i == j:
                # 对角线：直方图
                for region, rc in REGION_COLORS.items():
                    subset = df[df['地区'] == region][variables[i]]
                    ax.hist(subset, bins=8, alpha=0.6, color=rc, edgecolor='white')
                ax.set_facecolor('#f0f0f0')
            else:
                x = df[variables[j]]
                y = df[variables[i]]
                ax.scatter(x, y, c=colors, alpha=0.75, s=50, edgecolors='white', linewidths=0.5)

                # 回归线
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() > 2:
                    z = np.polyfit(x[mask], y[mask], 1)
                    p = np.poly1d(z)
                    xline = np.linspace(x.min(), x.max(), 100)
                    ax.plot(xline, p(xline), color='#d62728', lw=1.5, alpha=0.8)
                    # 相关系数
                    r = np.corrcoef(x[mask], y[mask])[0, 1]
                    ax.text(0.97, 0.97, f'r={r:.2f}', transform=ax.transAxes,
                            ha='right', va='top', fontsize=8.5,
                            color='#d62728' if abs(r) > 0.5 else '#666666',
                            fontweight='bold' if abs(r) > 0.5 else 'normal')

            # 轴标签
            if i == n - 1:
                ax.set_xlabel(variables[j], fontsize=8, labelpad=2)
            if j == 0:
                ax.set_ylabel(variables[i], fontsize=8, labelpad=2)
            ax.tick_params(labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#cccccc')

    # 图例
    legend_elems = [mpatches.Patch(color=REGION_COLORS[r], label=r) for r in ['东部', '中部', '西部', '东北']]
    fig.legend(handles=legend_elems, loc='upper right', fontsize=9,
               title='地区', title_fontsize=10, framealpha=0.85,
               bbox_to_anchor=(0.98, 0.98))

    fig.suptitle('GDP变化与产业结构变化的相关性分析（2015-2024，各省）\n'
                 '散点矩阵：对角线=分布直方图，非对角线=散点+回归线，r=相关系数',
                 fontsize=13, fontweight='bold', color='#222222', y=1.005)

    plt.tight_layout(rect=[0, 0, 1, 1])
    out = os.path.join(OUTPUT_DIR, '图表2_GDP与产业结构相关性散点矩阵.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'已保存：{out}')


# ============================================================
# 图表3：各省GDP发展数据的相关性（相关矩阵热力图）
# ============================================================
def chart3_gdp_corr_heatmap(df_province):
    """
    以年份为行，省份GDP为列，构建时序宽表，
    计算各省GDP时序之间的Pearson相关系数矩阵，
    绘制热力图，揭示发展趋势相似的省份群体。
    """
    pivot = df_province.pivot(index='年份', columns='省份', values='地区生产总值 (GDP)_亿元')

    # 检查是否有缺失数据
    missing_provs = pivot.columns[pivot.isna().any()].tolist()
    if missing_provs:
        print(f'警告：以下省份存在缺失数据: {missing_provs}')
    
    # 填充缺失值为该省份的均值（如果某年缺失）
    pivot = pivot.fillna(pivot.mean())
    
    # 再次检查并删除仍有NaN的列（如果省份全部数据都缺失）
    pivot = pivot.dropna(axis=1, how='all')

    # 按地区排序，使相似省份相邻
    order = [p for p in (EAST_CHINA + CENTRAL_CHINA + WEST_CHINA + NORTHEAST) if p in pivot.columns]
    pivot = pivot[order]

    corr = pivot.corr()

    # 确保对角线为1（消除浮点数误差）
    for i, prov in enumerate(order):
        corr.iloc[i, i] = 1.0

    # 四大地区分界线位置
    n_east = len([p for p in EAST_CHINA if p in order])
    n_central = len([p for p in CENTRAL_CHINA if p in order])
    n_west = len([p for p in WEST_CHINA if p in order])

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    cmap = plt.cm.RdYlGn
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    # 数值标注（显示绝对值>0.7的非对角格，对角线始终显示）
    n = len(order)
    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            if i == j:
                # 对角线：始终显示
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=5, color='#333333', fontweight='bold')
            elif abs(val) > 0.70:  # 降低阈值从0.85到0.70，显示更多相关系数
                color = 'white' if abs(val) > 0.92 else '#333333'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=5, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(order, rotation=90, fontsize=8)
    ax.set_yticklabels(order, fontsize=8)

    # 地区分界线
    boundaries = [0, n_east, n_east + n_central, n_east + n_central + n_west, n]
    region_names = ['东部', '中部', '西部', '东北']
    for idx, (b1, b2, rname) in enumerate(zip(boundaries[:-1], boundaries[1:], region_names)):
        mid = (b1 + b2 - 1) / 2
        ax.text(n + 0.3, mid, rname, ha='left', va='center', fontsize=8,
                color=list(REGION_COLORS.values())[idx], fontweight='bold')

    # 绘制边界线
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color='white', lw=2.5)
        ax.axvline(b - 0.5, color='white', lw=2.5)

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=25, pad=0.02)
    cbar.set_label('Pearson 相关系数', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title('各省GDP时序发展相关性矩阵（2015-2024）\n'
                 '颜色越绿=相关性越强（发展趋势相似），白线=六大地区分界，数字=相关系数≥0.70',
                 fontsize=12, fontweight='bold', color='#222222')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, '图表3_各省GDP发展相关性热力图.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'已保存：{out}')


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print('加载数据...')
    df_province, df_national = load_data()

    print('生成图表1：GDP构成旭日图...')
    chart1_sunburst(df_province)

    print('生成图表2：GDP与产业结构相关性散点矩阵...')
    chart2_correlation_scatter(df_province, df_national)

    print('生成图表3：各省GDP发展相关性热力图...')
    chart3_gdp_corr_heatmap(df_province)

    print('\n全部完成！')