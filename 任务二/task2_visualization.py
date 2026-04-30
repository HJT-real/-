"""
任务二：地区经济规模对比
生成5个静态图表，展示不同地区或省份之间的GDP规模差异
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.collections import PatchCollection
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

# 统一配色
INDUSTRY_COLORS = {
    '第一产业': '#2ca02c',
    '第二产业': '#1f77b4',
    '第三产业': '#ff7f0e',
}
COLOR_BG = '#fafafa'
COLOR_PRIMARY = '#1f77b4'
COLOR_ACCENT = '#d62728'

# 输出目录
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), '数据集')

# ============================================================
# 省份名称映射（CSV短名 → GeoJSON全称）
# ============================================================
PROVINCE_NAME_MAP = {
    '北京': '北京市', '天津': '天津市', '河北': '河北省', '山西': '山西省',
    '内蒙古': '内蒙古自治区', '辽宁': '辽宁省', '吉林': '吉林省', '黑龙江': '黑龙江省',
    '上海': '上海市', '江苏': '江苏省', '浙江': '浙江省', '安徽': '安徽省',
    '福建': '福建省', '江西': '江西省', '山东': '山东省', '河南': '河南省',
    '湖北': '湖北省', '湖南': '湖南省', '广东': '广东省', '广西': '广西壮族自治区',
    '海南': '海南省', '重庆': '重庆市', '四川': '四川省', '贵州': '贵州省',
    '云南': '云南省', '西藏': '西藏自治区', '陕西': '陕西省', '甘肃': '甘肃省',
    '青海': '青海省', '宁夏': '宁夏回族自治区', '新疆': '新疆维吾尔自治区',
}

# 反向映射（全称 → 短名）
PROVINCE_NAME_MAP_REV = {v: k for k, v in PROVINCE_NAME_MAP.items()}

# 全称 → 短名（用于图表显示，CSV中省级行政区列是全称）
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


# ============================================================
# 数据读取
# ============================================================
def load_data():
    """读取所有数据文件"""
    # 各省份GDP统计表（2015-2024）
    df_province = pd.read_csv(
        os.path.join(DATA_DIR, '2015-2024年各省份GDP统计表.csv'),
        encoding='utf-8-sig'
    )
    df_province.columns = df_province.columns.str.strip()

    # 2015和2024年各省GDP产业数据
    df_compare = pd.read_csv(
        os.path.join(DATA_DIR, '2015和2024年各省GDP产业数据.csv'),
        encoding='utf-8-sig'
    )
    df_compare.columns = df_compare.columns.str.strip()

    return df_province, df_compare


# ============================================================
# 图表1：2024年各省/地区GDP总量对比（水平柱状图）
# ============================================================
def chart1_province_gdp_2024(df_province):
    """2024年各省GDP总量对比 - 水平柱状图"""
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # 筛选2024年数据
    df_2024 = df_province[df_province['年份'] == 2024].copy()
    df_2024 = df_2024.sort_values('地区生产总值 (GDP)_亿元', ascending=True)

    provinces = df_2024['省级行政区'].values
    gdp = df_2024['地区生产总值 (GDP)_亿元'].values / 10000  # 万亿元

    # 使用短名
    short_names = [FULL_TO_SHORT.get(p, p) for p in provinces]

    # 颜色渐变
    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=min(gdp), vmax=max(gdp))
    colors = [cmap(norm(v)) for v in gdp]

    bars = ax.barh(range(len(provinces)), gdp, height=0.7, color=colors,
                   edgecolor='white', linewidth=0.5, zorder=3)

    # 标注GDP数值
    for i, (bar, val) in enumerate(zip(bars, gdp)):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', ha='left',
                fontsize=8, color='#333333', fontweight='bold')

    # 高亮前5名
    top5_idx = list(range(len(provinces) - 5, len(provinces)))
    for idx in top5_idx:
        bars[idx].set_edgecolor('#d62728')
        bars[idx].set_linewidth(1.5)

    ax.set_yticks(range(len(provinces)))
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel('GDP总量（万亿元）', fontsize=13, fontweight='bold')
    ax.set_title('2024年各省/地区GDP总量对比', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 标注前5名
    top5_provinces = short_names[-5:]
    top5_gdp = gdp[-5:]
    ax.text(0.98, 0.02,
            f'TOP5: {", ".join(top5_provinces[::-1])}',
            transform=ax.transAxes, fontsize=10, ha='right',
            color=COLOR_ACCENT, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff0f0',
                      edgecolor=COLOR_ACCENT, alpha=0.9))

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, '图表1_2024各省GDP总量对比.png')
    plt.savefig(filepath)
    plt.close()
    print(f'[OK] 图表1已保存: {filepath}')


# ============================================================
# 图表2：2024年各省/地区GDP产业构成对比（堆叠水平柱状图）
# ============================================================
def chart2_province_industry_2024(df_province):
    """2024年各省GDP产业构成对比 - 堆叠水平柱状图"""
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # 筛选2024年数据
    df_2024 = df_province[df_province['年份'] == 2024].copy()
    df_2024 = df_2024.sort_values('地区生产总值 (GDP)_亿元', ascending=True)

    provinces = df_2024['省级行政区'].values
    short_names = [FULL_TO_SHORT.get(p, p) for p in provinces]

    p1 = df_2024['第一产业占 GDP 比重_%'].values
    p2 = df_2024['第二产业占 GDP 比重_%'].values
    p3 = df_2024['第三产业占 GDP 比重_%'].values

    y_pos = range(len(provinces))

    # 堆叠水平柱状图
    ax.barh(y_pos, p1, height=0.7, label='第一产业',
            color=INDUSTRY_COLORS['第一产业'], edgecolor='white', linewidth=0.5)
    ax.barh(y_pos, p2, height=0.7, left=p1, label='第二产业',
            color=INDUSTRY_COLORS['第二产业'], edgecolor='white', linewidth=0.5)
    ax.barh(y_pos, p3, height=0.7, left=p1 + p2, label='第三产业',
            color=INDUSTRY_COLORS['第三产业'], edgecolor='white', linewidth=0.5)

    # 标注第三产业占比
    for i, val in enumerate(p3):
        if val > 15:  # 只标注足够宽的部分
            ax.text(p1[i] + p2[i] + val / 2, i, f'{val:.1f}%',
                    ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    # 标注第二产业占比
    for i, val in enumerate(p2):
        if val > 15:
            ax.text(p1[i] + val / 2, i, f'{val:.1f}%',
                    ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel('产业占比（%）', fontsize=13, fontweight='bold')
    ax.set_title('2024年各省/地区GDP产业构成对比', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9,
              edgecolor='gray', fancybox=True)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, '图表2_2024各省GDP产业构成对比.png')
    plt.savefig(filepath)
    plt.close()
    print(f'[OK] 图表2已保存: {filepath}')


# ============================================================
# 图表3：各省/地区2015与2024两年经济发展对比图（哑铃图）
# ============================================================
def chart3_2015_vs_2024(df_province):
    """各省2015与2024两年GDP对比 - 哑铃图"""
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # 筛选2015和2024年数据
    df_2015 = df_province[df_province['年份'] == 2015][['省级行政区', '地区生产总值 (GDP)_亿元']].copy()
    df_2024 = df_province[df_province['年份'] == 2024][['省级行政区', '地区生产总值 (GDP)_亿元']].copy()

    df_merge = pd.merge(df_2015, df_2024, on='省级行政区', suffixes=('_2015', '_2024'))
    df_merge['增长量'] = df_merge['地区生产总值 (GDP)_亿元_2024'] - df_merge['地区生产总值 (GDP)_亿元_2015']
    df_merge['增长率'] = df_merge['增长量'] / df_merge['地区生产总值 (GDP)_亿元_2015'] * 100
    df_merge = df_merge.sort_values('地区生产总值 (GDP)_亿元_2024', ascending=True)

    provinces = df_merge['省级行政区'].values
    short_names = [FULL_TO_SHORT.get(p, p) for p in provinces]
    gdp_2015 = df_merge['地区生产总值 (GDP)_亿元_2015'].values / 10000
    gdp_2024 = df_merge['地区生产总值 (GDP)_亿元_2024'].values / 10000
    growth_rate = df_merge['增长率'].values

    y_pos = range(len(provinces))

    # 连接线（哑铃的杆）
    for i in range(len(provinces)):
        ax.plot([gdp_2015[i], gdp_2024[i]], [i, i],
                color='#cccccc', linewidth=2, zorder=1)

    # 2015年点
    ax.scatter(gdp_2015, y_pos, color='#88bbdd', s=60, zorder=3,
               edgecolors='white', linewidths=1, label='2015年')
    # 2024年点
    ax.scatter(gdp_2024, y_pos, color=COLOR_PRIMARY, s=80, zorder=4,
               edgecolors='white', linewidths=1, label='2024年')

    # 标注增长率
    for i, rate in enumerate(growth_rate):
        color = COLOR_ACCENT if rate > 100 else '#666666'
        ax.text(max(gdp_2015[i], gdp_2024[i]) + 0.15, i,
                f'+{rate:.1f}%', va='center', fontsize=8,
                color=color, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel('GDP总量（万亿元）', fontsize=13, fontweight='bold')
    ax.set_title('各省/地区2015与2024年GDP对比\n（哑铃图：左端2015年，右端2024年，标注增长率）',
                 fontsize=15, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9,
              edgecolor='gray', fancybox=True)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, '图表3_2015与2024各省GDP对比.png')
    plt.savefig(filepath)
    plt.close()
    print(f'[OK] 图表3已保存: {filepath}')


# ============================================================
# 图表4：基于地图的2024年各省/地区GDP总量对比（Choropleth地图）
# ============================================================
def chart4_gdp_choropleth_map(df_province):
    """基于地图的2024年各省GDP总量对比"""
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # 读取GeoJSON
    gdf = gpd.read_file(os.path.join(DATA_DIR, 'china_provinces.geojson'))

    # 筛选2024年数据
    df_2024 = df_province[df_province['年份'] == 2024].copy()

    # CSV的省级行政区列已经是全称（如"北京市"），直接用于匹配
    # 合并地理数据与GDP数据
    gdf_merged = gdf.merge(df_2024, left_on='name', right_on='省级行政区', how='left')

    # 排除台湾、香港、澳门和南海诸岛
    exclude = ['台湾省', '香港特别行政区', '澳门特别行政区']
    gdf_merged = gdf_merged[~gdf_merged['name'].isin(exclude)]

    # 绘制地图
    gdf_merged.plot(
        column='地区生产总值 (GDP)_亿元',
        cmap='Blues',
        linewidth=0.8,
        edgecolor='white',
        legend=True,
        ax=ax,
        legend_kwds={
            'label': 'GDP（亿元）',
            'orientation': 'horizontal',
            'shrink': 0.6,
            'aspect': 30,
            'pad': 0.05,
        },
        missing_kwds={
            'color': '#eeeeee',
            'edgecolor': '#cccccc',
            'label': '无数据'
        }
    )

    # 标注省份简称和GDP（标注所有省份）
    # 为地理位置接近的省份添加偏移量，避免重叠
    label_offset = {
        '北京市': (0.8, 0.8),    # 北京向右上偏移
        '天津市': (0.8, -0.5),   # 天津向右下偏移
        '河北省': (-0.5, 0.3),   # 河北向左上偏移
        '上海市': (0.5, 0.5),     # 上海向右上偏移
        '江苏省': (-0.5, 0.5),    # 江苏向左上偏移
        '浙江省': (0.5, -0.5),    # 浙江向右下偏移
        '安徽省': (-0.5, -0.5),   # 安徽向左下偏移
        '香港特别行政区': (0.5, -0.5),
        '澳门特别行政区': (-0.5, -0.5),
    }

    for idx, row in gdf_merged.iterrows():
        if pd.notna(row.get('地区生产总值 (GDP)_亿元')):
            # 获取省份中心点
            centroid = row.geometry.centroid
            short_name = FULL_TO_SHORT.get(row['name'], row['name'])
            gdp_val = row['地区生产总值 (GDP)_亿元'] / 10000

            # 根据GDP大小调整显示格式
            if gdp_val >= 1.0:
                gdp_text = f'{gdp_val:.1f}万亿'
            else:
                gdp_text = f'{row["地区生产总值 (GDP)_亿元"]:.0f}亿'

            # 应用偏移量
            offset_x, offset_y = label_offset.get(row['name'], (0, 0))
            xy_pos = (centroid.x + offset_x, centroid.y + offset_y)

            ax.annotate(
                f'{short_name}\n{gdp_text}',
                xy=xy_pos,
                fontsize=6, fontweight='bold',
                color='#333333', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                          edgecolor='#cccccc', alpha=0.85)
            )

    ax.set_title('2024年各省/地区GDP总量分布图', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(72, 138)
    ax.set_ylim(15, 55)
    ax.axis('off')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, '图表4_2024各省GDP地图.png')
    plt.savefig(filepath)
    plt.close()
    print(f'[OK] 图表4已保存: {filepath}')


# ============================================================
# 图表5：基于变形地图的各省GDP发展曲线可视化（圆形气泡形式）
# ============================================================
def chart5_cartogram_sparklines(df_province):
    """基于变形地图（Dorling Cartogram圆形气泡）的各省GDP发展曲线可视化"""
    from matplotlib.patches import Circle as MplCircle
    
    # 定义省份在地图上的近似坐标位置（经纬度，用于布局）
    # 格式：(经度, 纬度) -> 将转换为画布坐标
    map_positions = {
        # 东北
        '黑龙江': (128, 50), '吉林': (126, 44), '辽宁': (123, 40),
        # 华北
        '内蒙古': (110, 45),
        '北京':   (120, 42), '天津': (126, 39),
        '河北':   (116, 38), '山西': (108, 38),
        # 西北
        '新疆':   (76, 42), '甘肃': (96, 38), '宁夏': (103, 36), '陕西': (108, 34),
        '青海':   (96, 34),
        # 西藏
        '西藏':   (86, 31),
        # 西南
        '四川':   (103, 30), '重庆': (108, 29),
        '云南':   (100, 24), '贵州': (106, 27),
        # 华中
        '河南':   (113, 34), '湖北': (112, 30),
        '湖南':   (111, 27),
        # 华东（长三角向右拉开）
        '山东':   (119, 36),
        '安徽':   (117, 32), '江苏': (122, 33),
        '上海':   (128, 31), '浙江': (123, 28),
        '江西':   (116, 27), '福建': (119, 24),
        # 华南
        '广东':   (114, 22), '广西': (108, 23), '海南': (110, 19),
    }

    fig, ax = plt.subplots(figsize=(22, 16))
    fig.patch.set_facecolor('#f5f7fa')
    ax.set_facecolor('#f5f7fa')

    # 准备各省2015-2024年GDP数据
    df_all = df_province.copy()

    # 计算每个省的GDP用于气泡大小
    gdp_2024 = {}
    for prov in map_positions.keys():
        prov_data = df_all[df_all['省级行政区'].str.contains(prov)]
        if len(prov_data) > 0:
            row_2024 = prov_data[prov_data['年份'] == 2024]
            if len(row_2024) > 0:
                gdp_2024[prov] = row_2024['地区生产总值 (GDP)_亿元'].values[0]

    # 准备GDP序列用于折线
    gdp_series_dict = {}
    for prov in map_positions.keys():
        prov_data = df_all[df_all['省级行政区'].str.contains(prov)].sort_values('年份')
        if len(prov_data) > 0:
            gdp_series_dict[prov] = {
                'years': prov_data['年份'].values,
                'values': prov_data['地区生产总值 (GDP)_亿元'].values
            }

    # 颜色映射
    if gdp_2024:
        cmap = plt.cm.YlGnBu
        vmin = min(gdp_2024.values())
        vmax = max(gdp_2024.values())
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # 画布范围设置 - 扩大范围以容纳气泡
    ax.set_xlim(68, 140)
    ax.set_ylim(10, 56)
    ax.set_aspect('equal')

    # 气泡大小映射 - 减小最大半径避免重叠
    min_radius = 1.7
    max_radius = 4.5  # 缩小半径，配合新坐标间距避免重叠
    if gdp_2024:
        gdp_values = list(gdp_2024.values())
        gdp_max = max(gdp_values)
        gdp_min = min(gdp_values)

    # 预计算所有省份半径，供排斥算法使用
    radii_dict = {}
    for prov in map_positions:
        if prov in gdp_2024:
            gdp_val = gdp_2024[prov]
            ratio = np.sqrt((gdp_val - gdp_min) / (gdp_max - gdp_min)) if gdp_max > gdp_min else 0.5
            radii_dict[prov] = min_radius + ratio * (max_radius - min_radius)
        else:
            radii_dict[prov] = min_radius

    # 力导向排斥：迭代推开重叠圆，直到所有圆间距满足要求
    resolved_pos = {k: np.array(v, dtype=float) for k, v in map_positions.items()}
    prov_keys = list(resolved_pos.keys())
    gap = 0.2  # 圆与圆之间的最小间隙（数据单位）
    for _ in range(3000):
        max_overlap = 0.0
        for i in range(len(prov_keys)):
            for j in range(i + 1, len(prov_keys)):
                k1, k2 = prov_keys[i], prov_keys[j]
                diff = resolved_pos[k2] - resolved_pos[k1]
                dist = float(np.linalg.norm(diff))
                min_dist = radii_dict[k1] + radii_dict[k2] + gap
                if dist < min_dist:
                    overlap = min_dist - dist
                    max_overlap = max(max_overlap, overlap)
                    if dist < 1e-6:
                        diff = np.array([1.0, 0.0])
                        dist = 1.0
                    push = (overlap / 2 + 0.02) * diff / dist
                    resolved_pos[k1] -= push
                    resolved_pos[k2] += push
        if max_overlap < 0.02:
            break

    for prov in map_positions:
        lon, lat = resolved_pos[prov]
        radius = radii_dict[prov]

        if prov in gdp_2024:
            color = cmap(norm(gdp_2024[prov]))
        else:
            color = '#cccccc'

        # 绘制圆形气泡
        circle = MplCircle((lon, lat), radius,
                          facecolor=color, edgecolor='white',
                          linewidth=1.5, alpha=0.85, zorder=2)
        ax.add_patch(circle)

        # 在圆形内上半部分标注省份名称
        font_size = max(12, min(18, radius * 2.7))
        name_y = lat + radius * 0.35
        ax.text(lon, name_y, prov,
               ha='center', va='center',
               fontsize=font_size, fontweight='bold',
               color='white', zorder=3,
               path_effects=[
                   __import__('matplotlib.patheffects', fromlist=['withStroke']).withStroke(
                       linewidth=3, foreground='#1a1a2e')
               ])

        # 在圆形内下半部分绘制发展曲线（所有省份均绘制）
        if prov in gdp_series_dict and len(gdp_series_dict[prov]['values']) > 1:
            series = gdp_series_dict[prov]
            values = series['values']

            v_min, v_max = values.min(), values.max()
            norm_vals = (values - v_min) / (v_max - v_min) if v_max > v_min else np.ones_like(values) * 0.5

            # 曲线区域：圆心下方，纵向占半径的 60%，横向占半径的 80%
            curve_h = radius * 0.55   # 曲线区域高度
            curve_w = radius * 0.80   # 曲线区域半宽
            curve_cy = lat - radius * 0.30  # 曲线区域中心 y（偏下）

            n_pts = len(values)
            x_pts = np.linspace(lon - curve_w, lon + curve_w, n_pts)
            y_pts = curve_cy - curve_h * 0.5 + norm_vals * curve_h

            lw = max(0.6, min(1.2, radius * 0.3))
            ax.plot(x_pts, y_pts, color='#8b0000', linewidth=lw + 0.6,
                   alpha=0.6, zorder=4)  # 深红描边提升对比
            ax.plot(x_pts, y_pts, color='#ff6b6b', linewidth=lw,
                   alpha=1.0, zorder=5)
            # 起点（浅粉）和终点（深红）标记
            ms = max(1.5, radius * 0.6)
            ax.plot(x_pts[0], y_pts[0], 'o', color='#ffb3b3', markersize=ms, zorder=6)
            ax.plot(x_pts[-1], y_pts[-1], 'o', color='#cc0000', markersize=ms, zorder=6)

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                        shrink=0.4, aspect=25, pad=0.08)
    cbar.set_label('2024年GDP（亿元）- 气泡面积表示GDP规模', fontsize=11, fontweight='bold')

    ax.set_title('基于变形地图的各省GDP发展曲线可视化（Dorling Cartogram圆形气泡图）\n气泡面积与GDP成正比，位置近似中国地图布局',
                 fontsize=13, fontweight='bold', pad=15, color='#1a1a2e')
    ax.axis('off')

    # 图例说明
    ax.text(0.5, 0.02,
            '● 圆形面积表示GDP规模（平方根比例）  |  红色折线表示2015-2024年发展趋势  |  位置近似中国地图布局',
            transform=ax.transAxes, fontsize=10, color='#444444',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#cccccc', alpha=0.95))

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, '图表5_变形地图GDP发展曲线.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[OK] 图表5已保存: {filepath}')


# ============================================================
# 主程序
# ============================================================
def main():
    print('=' * 60)
    print('任务二：地区经济规模对比 - 图表生成')
    print('=' * 60)

    # 读取数据
    df_province, df_compare = load_data()
    print(f'[INFO] 数据加载完成:')
    print(f'  - 各省份GDP统计表: {len(df_province)} 条记录')
    print(f'  - 2015/2024对比数据: {len(df_compare)} 条记录')
    print()

    # 生成图表
    chart1_province_gdp_2024(df_province)
    chart2_province_industry_2024(df_province)
    chart3_2015_vs_2024(df_province)
    chart4_gdp_choropleth_map(df_province)
    chart5_cartogram_sparklines(df_province)

    print('\n' + '=' * 60)
    print('[DONE] 任务二全部5个图表生成完毕！')
    print('=' * 60)


if __name__ == '__main__':
    main()
