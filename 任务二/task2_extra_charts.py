"""
任务二补充图表（方案E、J、F、K、L）
图表6：省份GDP排名斜率图
图表7：南丁格尔玫瑰图
图表8：各省GDP份额堆叠面积图
图表9：GDP增量瀑布图
图表10：图表10_2024年各省GDP总量与人均GDP关系散点图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import geopandas as gpd
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

# 省份全称 → 短名
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

EAST_CHINA   = ['北京', '天津', '河北', '上海', '江苏', '浙江', '福建', '山东', '广东', '海南']
CENTRAL_CHINA = ['山西', '安徽', '江西', '河南', '湖北', '湖南']
WEST_CHINA   = ['内蒙古', '广西', '重庆', '四川', '贵州', '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆']
NORTHEAST    = ['辽宁', '吉林', '黑龙江']

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

    df_compare = pd.read_csv(
        os.path.join(DATA_DIR, '2015和2024年各省GDP产业数据.csv'),
        encoding='utf-8-sig'
    )
    df_compare.columns = df_compare.columns.str.strip()

    geo_path = os.path.join(DATA_DIR, 'china_provinces.geojson')
    gdf = gpd.read_file(geo_path)

    return df_province, df_compare, gdf


# ============================================================
# 图表6：省份GDP排名斜率图（方案E）
# ============================================================
def chart6_slope(df_province):
    df_2015 = df_province[df_province['年份'] == 2015][['省份', '地区生产总值 (GDP)_亿元']].copy()
    df_2024 = df_province[df_province['年份'] == 2024][['省份', '地区生产总值 (GDP)_亿元']].copy()

    df_2015['rank2015'] = df_2015['地区生产总值 (GDP)_亿元'].rank(ascending=False).astype(int)
    df_2024['rank2024'] = df_2024['地区生产总值 (GDP)_亿元'].rank(ascending=False).astype(int)

    df = df_2015.merge(df_2024, on='省份', suffixes=('_2015', '_2024'))
    df['change'] = df['rank2015'] - df['rank2024']  # 正=排名上升

    fig, ax = plt.subplots(figsize=(10, 12))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    n = 31
    y_2015 = {row['省份']: row['rank2015'] for _, row in df.iterrows()}
    y_2024 = {row['省份']: row['rank2024'] for _, row in df.iterrows()}

    for _, row in df.iterrows():
        prov = row['省份']
        r15 = row['rank2015']
        r24 = row['rank2024']
        ch = row['change']
        if ch > 0:
            color = COLOR_ACCENT
        elif ch < 0:
            color = COLOR_PRIMARY
        else:
            color = '#aaaaaa'
        lw = 1.0 + abs(ch) * 0.25
        ax.plot([0, 1], [r15, r24], color=color, lw=lw, alpha=0.8, solid_capstyle='round')

    for _, row in df.iterrows():
        prov = row['省份']
        r15 = row['rank2015']
        r24 = row['rank2024']
        ch = row['change']
        color = COLOR_ACCENT if ch > 0 else (COLOR_PRIMARY if ch < 0 else '#666666')
        ax.text(-0.03, r15, f'{r15}  {prov}', ha='right', va='center', fontsize=8.5, color='#333333')
        arrow = ''
        if ch > 2:
            arrow = f' ↑{ch}'
        elif ch < -2:
            arrow = f' ↓{abs(ch)}'
        ax.text(1.03, r24, f'{prov}  {r24}{arrow}', ha='left', va='center', fontsize=8.5, color=color)

    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(n + 0.3, -0.2)  # 上边界收紧，减少顶部空白
    ax.axis('off')

    ax.legend(loc='lower center', ncol=3, fontsize=9,
              framealpha=0.8, bbox_to_anchor=(0.5, 0.01))

    ax.set_title('各省GDP总量排名变化（2015 → 2024）\n斜率图：连线方向反映排名升降，线宽反映变化幅度',
                 fontsize=13, fontweight='bold', color='#222222', pad=8)

    out = os.path.join(OUTPUT_DIR, '图表6_省份GDP排名斜率图.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'已保存：{out}')


# ============================================================
# 图表7：南丁格尔玫瑰图（方案J）
# ============================================================
def chart7_nightingale(df_province):
    df_2024 = df_province[df_province['年份'] == 2024].copy()
    df_2024 = df_2024.sort_values('地区生产总值 (GDP)_亿元', ascending=False)
    df_2024['region'] = df_2024['省份'].apply(get_region)

    provinces = df_2024['省份'].values
    gdp = df_2024['地区生产总值 (GDP)_亿元'].values
    regions = df_2024['region'].values

    n = len(provinces)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n * 0.85

    # 用平方根缩放半径，避免广东过大
    radii = np.sqrt(gdp / gdp.max())

    fig, ax = plt.subplots(figsize=(13, 13), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    color_map = {r: REGION_COLORS[r] for r in REGION_COLORS}
    bar_colors = [color_map[r] for r in regions]

    bars = ax.bar(angles, radii, width=width, bottom=0.05,
                  color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.8)

    # 省份名标注
    for angle, radius, prov, gdp_val in zip(angles, radii, provinces, gdp):
        label_r = radius + 0.08
        rotation = np.degrees(angle)
        if np.pi / 2 < angle < 3 * np.pi / 2:
            rotation += 180
            ha = 'right'
        else:
            ha = 'left'
        ax.text(angle, label_r, prov, rotation=rotation, rotation_mode='anchor',
                ha=ha, va='center', fontsize=7.5, color='#333333')

    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    legend_elems = [mpatches.Patch(color=REGION_COLORS[r], label=r) for r in ['东部', '中部', '西部', '东北']]
    ax.legend(handles=legend_elems, loc='lower right', fontsize=9, framealpha=0.85,
              bbox_to_anchor=(1.15, 0.05))

    # 中心标注
    ax.text(0, 0, '2024年\n各省GDP\n规模', ha='center', va='center', fontsize=11,
            fontweight='bold', color='#333333')

    ax.set_title('2024年各省GDP规模南丁格尔玫瑰图\n扇形半径∝√GDP（开方缩放），颜色区分四大地区',
                 fontsize=13, fontweight='bold', color='#222222', y=1.05)

    out = os.path.join(OUTPUT_DIR, '图表7_南丁格尔玫瑰图.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'已保存：{out}')


# ============================================================
# 图表8：各省GDP份额堆叠面积图（方案F）
# ============================================================
def chart8_share_area(df_province):
    years = sorted(df_province['年份'].unique())

    national = df_province.groupby('年份')['地区生产总值 (GDP)_亿元'].sum()

    pivot = df_province.pivot(index='年份', columns='省份', values='地区生产总值 (GDP)_亿元')
    share = pivot.div(national, axis=0) * 100

    # 四大地区堆叠顺序：东部在下，中部，东北，西部在上
    region_order = [EAST_CHINA, CENTRAL_CHINA, NORTHEAST, WEST_CHINA]
    region_labels = ['东部', '中部', '东北', '西部']
    region_cmaps = [plt.cm.Reds, plt.cm.Greens, plt.cm.Blues, plt.cm.Purples]

    order = []
    for grp in region_order:
        order += [p for p in grp if p in share.columns]
    share = share[order]

    def get_colors_for(names, cmap):
        idxs = np.linspace(0.35, 0.85, len(names))
        return [cmap(i) for i in idxs]

    colors = []
    for grp, cmap in zip(region_order, region_cmaps):
        names = [p for p in grp if p in share.columns]
        colors += get_colors_for(names, cmap)

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    ax.stackplot(years, [share[p].values for p in order], labels=order,
                 colors=colors, alpha=0.9)

    # 关键省份内标注
    cumsum = share.cumsum(axis=1)
    highlight = ['广东', '江苏', '山东', '浙江', '四川']
    for prov in highlight:
        if prov not in cumsum.columns:
            continue
        mid_year_idx = len(years) // 2
        mid_year = years[mid_year_idx]
        top = cumsum.loc[mid_year, prov]
        bottom = top - share.loc[mid_year, prov]
        mid_y = (top + bottom) / 2
        ax.text(mid_year, mid_y, prov, ha='center', va='center',
                fontsize=8, color='white', fontweight='bold', alpha=0.9)

    # 东部合计份额折线（次Y轴）
    east_provs = [p for p in EAST_CHINA if p in share.columns]
    east_share = share[east_provs].sum(axis=1)
    ax2 = ax.twinx()
    ax2.plot(years, east_share, color='#d62728', lw=2, ls='--', marker='o',
             markersize=5, label='东部合计占比', zorder=5)
    ax2.set_ylabel('东部合计占全国GDP比重 (%)', fontsize=10, color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.set_ylim(0, 70)
    ax2.legend(loc='upper right', fontsize=9)

    ax.set_xlim(years[0], years[-1])
    ax.set_ylim(0, 100)
    ax.set_xlabel('年份', fontsize=11)
    ax.set_ylabel('占全国GDP比重 (%)', fontsize=11)
    ax.set_xticks(years)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))
    ax.grid(axis='y', alpha=0.3, ls='--')

    legend_elems = [
        mpatches.Patch(color=cmap(0.6), label=lbl)
        for lbl, cmap in zip(region_labels, region_cmaps)
    ]
    ax.legend(handles=legend_elems, loc='lower left', fontsize=9, framealpha=0.85, ncol=2)

    ax.set_title('2015-2024年各省GDP份额演变\n堆叠面积图：颜色区分四大地区，红虚线=东部合计占比',
                 fontsize=13, fontweight='bold', color='#222222')

    out = os.path.join(OUTPUT_DIR, '图表8_各省GDP份额演变.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'已保存：{out}')


# ============================================================
# 图表9：GDP增量瀑布图（方案K）
# ============================================================
def chart9_waterfall(df_province):
    df_2015 = df_province[df_province['年份'] == 2015][['省份', '地区生产总值 (GDP)_亿元']].set_index('省份')
    df_2024 = df_province[df_province['年份'] == 2024][['省份', '地区生产总值 (GDP)_亿元']].set_index('省份')

    inc = (df_2024['地区生产总值 (GDP)_亿元'] - df_2015['地区生产总值 (GDP)_亿元']).dropna()
    inc = inc.sort_values(ascending=False)

    provinces = list(inc.index)
    values = inc.values / 10000  # 万亿元
    total = values.sum()

    # 累计占比
    cumulative = np.cumsum(values) / total * 100

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # 修改1：使用 Blues 配色（大值深色，小值浅色），避免广东等大值柱子颜色过浅看不见
    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    colors = [cmap(norm(v)) for v in values]

    x = np.arange(len(provinces))
    bars = ax.bar(x, values, color=colors, edgecolor='white', linewidth=0.5, zorder=3)

    # 修改2：所有柱子顶部都显示数值（之前只显示前10名）
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7.5, color='#333333', fontweight='bold')

    # 全国合计柱（右侧独立）
    total_x = len(provinces) + 0.5
    ax.bar(total_x, total, color='#d62728', alpha=0.85, edgecolor='white', linewidth=0.8, zorder=3, width=0.8)
    ax.text(total_x, total + 0.02, f'{total:.2f}\n(全国合计)', ha='center', va='bottom',
            fontsize=8.5, color='#d62728', fontweight='bold')

    # 累计折线（次Y轴）
    ax2 = ax.twinx()
    ax2.plot(x, cumulative, color='#ff7f0e', lw=2, marker='o', markersize=4,
             label='累计增量占比 (%)', zorder=5)
    # 标注80%线
    idx_80 = np.searchsorted(cumulative, 80)
    ax2.axhline(80, color='#ff7f0e', ls='--', lw=1, alpha=0.6)
    ax2.text(idx_80 + 0.3, 81, f'前{idx_80 + 1}省贡献80%', fontsize=8, color='#ff7f0e')
    ax2.set_ylabel('累计增量占比 (%)', fontsize=10, color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.set_ylim(0, 110)
    ax2.legend(loc='center right', fontsize=9)

    # 隐藏黑框：仅保留底部和左侧参考线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(provinces, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('GDP增量（万亿元）', fontsize=11)
    ax.set_xlim(-0.5, total_x + 0.8)
    ax.grid(axis='y', alpha=0.3, ls='--')

    ax.set_title('2015-2024年各省GDP增量贡献（万亿元）\n蓝色柱=省级增量（深色≥多），红柱=全国合计，橙线=累计贡献占比',
                 fontsize=13, fontweight='bold', color='#222222')

    out = os.path.join(OUTPUT_DIR, '图表9_GDP增量瀑布图.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'已保存：{out}')


# ============================================================
# 图表10：GDP总量与人均GDP双维散点图
# ============================================================
def chart10_scatter_gdp(df_province, df_compare):
    df_2024 = df_province[df_province['年份'] == 2024][['省份', '地区生产总值 (GDP)_亿元']].copy()

    df_percap = df_compare[df_compare['年份'] == 2024][['地区', '人均地区生产总值']].copy()
    df_percap.columns = ['省份', '人均GDP']

    df = df_2024.merge(df_percap, on='省份', how='inner').dropna()
    df['地区'] = df['省份'].apply(get_region)
    df['GDP万亿'] = df['地区生产总值 (GDP)_亿元'] / 10000
    df['人均GDP万元'] = df['人均GDP'] / 10000

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    region_order = ['东部', '中部', '西部', '东北']
    for region in region_order:
        sub = df[df['地区'] == region]
        ax.scatter(sub['GDP万亿'], sub['人均GDP万元'],
                   s=120, color=REGION_COLORS[region], edgecolors='white',
                   linewidths=1.2, zorder=4, label=region, alpha=0.9)

    # 全部省份标注
    for _, row in df.iterrows():
        ax.annotate(
            row['省份'],
            xy=(row['GDP万亿'], row['人均GDP万元']),
            xytext=(4, 4), textcoords='offset points',
            fontsize=7.5, color='#333333',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.6, ec='none'),
        )

    # 均值参考线
    mean_gdp = df['GDP万亿'].mean()
    mean_percap = df['人均GDP万元'].mean()
    ax.axvline(mean_gdp, color='#888888', ls='--', lw=1, alpha=0.6)
    ax.axhline(mean_percap, color='#888888', ls='--', lw=1, alpha=0.6)
    ax.text(mean_gdp + 0.05, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1,
            f'均值{mean_gdp:.1f}万亿', fontsize=7.5, color='#888888', va='bottom')
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 0.05, mean_percap + 0.05,
            f'均值{mean_percap:.1f}万元', fontsize=7.5, color='#888888', va='bottom')

    # 四象限标注
    xmax = df['GDP万亿'].max() * 1.05
    ymax = df['人均GDP万元'].max() * 1.05
    ax.text(xmax * 0.75, ymax * 0.95, '高总量·高人均', fontsize=8,
            color='#aaaaaa', ha='right', va='top', style='italic')
    ax.text(xmax * 0.05, ymax * 0.95, '低总量·高人均', fontsize=8,
            color='#aaaaaa', ha='left', va='top', style='italic')
    ax.text(xmax * 0.75, mean_percap * 0.15, '高总量·低人均', fontsize=8,
            color='#aaaaaa', ha='right', va='bottom', style='italic')
    ax.text(xmax * 0.05, mean_percap * 0.15, '低总量·低人均', fontsize=8,
            color='#aaaaaa', ha='left', va='bottom', style='italic')

    ax.set_xlabel('GDP总量（万亿元）', fontsize=12, fontweight='bold')
    ax.set_ylabel('人均GDP（万元/人）', fontsize=12, fontweight='bold')
    ax.legend(title='地区', fontsize=9, title_fontsize=9,
              framealpha=0.85, loc='upper left')
    ax.grid(alpha=0.25, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title('图表10_2024年各省GDP总量与人均GDP关系散点图\n颜色区分四大地区，虚线为全国均值',
                 fontsize=13, fontweight='bold', color='#222222')

    out = os.path.join(OUTPUT_DIR, '图表10_2024年各省GDP总量与人均GDP关系散点图.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'已保存：{out}')


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print('加载数据...')
    df_province, df_compare, gdf = load_data()

    print('生成图表6：省份GDP排名斜率图...')
    chart6_slope(df_province)

    print('生成图表7：南丁格尔玫瑰图...')
    chart7_nightingale(df_province)

    print('生成图表8：各省GDP份额堆叠面积图...')
    chart8_share_area(df_province)

    print('生成图表9：GDP增量瀑布图...')
    chart9_waterfall(df_province)

    print('生成图表10：GDP总量与人均GDP散点图...')
    chart10_scatter_gdp(df_province, df_compare)

    print('\n全部完成！')