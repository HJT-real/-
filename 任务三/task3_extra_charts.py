"""
任务三补充图表（方案A、F、B、3D-A、3D-C）
图表4：平行坐标图——多维省份画像
图表5：桑基图——GDP增量来源与产业流向
图表6：PCA双标图——省份发展模式聚类
图表7：3D曲面图——省份×年份×GDP地形图
图表8：3D散点图——人均GDP×增速×第三产业占比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
    df_prov = pd.read_csv(
        os.path.join(DATA_DIR, '2015-2024年各省份GDP统计表.csv'), encoding='utf-8-sig')
    df_prov.columns = df_prov.columns.str.strip()
    df_prov['省份'] = df_prov['省级行政区'].map(FULL_TO_SHORT)

    df_detail = pd.read_csv(
        os.path.join(DATA_DIR, '2015和2024年各省GDP产业数据.csv'), encoding='utf-8-sig')
    df_detail.columns = df_detail.columns.str.strip()

    return df_prov, df_detail


# ============================================================
# 图表4：平行坐标图
# ============================================================
def chart4_parallel(df_prov, df_detail):
    df_2015 = df_prov[df_prov['年份'] == 2015].set_index('省份')
    df_2024 = df_prov[df_prov['年份'] == 2024].set_index('省份')
    percap_2024 = df_detail[df_detail['年份'] == 2024].set_index('地区')['人均地区生产总值']

    provinces = df_2024.index.tolist()

    df = pd.DataFrame(index=provinces)
    df['GDP总量(万亿)']      = df_2024['地区生产总值 (GDP)_亿元'] / 10000
    df['GDP增速(%)']        = (df_2024['地区生产总值 (GDP)_亿元'] - df_2015['地区生产总值 (GDP)_亿元']) \
                               / df_2015['地区生产总值 (GDP)_亿元'] * 100
    df['第三产业占比(%)']    = df_2024['第三产业占 GDP 比重_%']
    df['人均GDP(万元)']      = percap_2024.reindex(provinces) / 10000
    df['地区']               = [get_region(p) for p in provinces]
    df = df.dropna()

    axes_cols = ['GDP总量(万亿)', 'GDP增速(%)', '第三产业占比(%)', '人均GDP(万元)']
    n_axes = len(axes_cols)

    # 归一化到 [0, 1]
    norm_df = df[axes_cols].copy()
    for col in axes_cols:
        vmin, vmax = norm_df[col].min(), norm_df[col].max()
        norm_df[col] = (norm_df[col] - vmin) / (vmax - vmin)

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    x_pos = np.arange(n_axes)

    for prov, row in norm_df.iterrows():
        region = df.loc[prov, '地区']
        color  = REGION_COLORS[region]
        vals   = row[axes_cols].values
        # 高亮前5 GDP
        lw    = 2.5 if df.loc[prov, 'GDP总量(万亿)'] > 5 else 1.0
        alpha = 0.85 if df.loc[prov, 'GDP总量(万亿)'] > 5 else 0.45
        ax.plot(x_pos, vals, color=color, lw=lw, alpha=alpha, solid_capstyle='round')
        # 大省标注
        if df.loc[prov, 'GDP总量(万亿)'] > 5:
            ax.text(x_pos[-1] + 0.05, vals[-1], prov,
                    va='center', fontsize=8, color=color, fontweight='bold')

    # 轴线
    for i, col in enumerate(axes_cols):
        ax.axvline(i, color='#bbbbbb', lw=1.2, zorder=0)
        orig = df[col]
        # 刻度：min / mid / max
        for frac, label in [(0, f'{orig.min():.1f}'), (0.5, f'{orig.mean():.1f}'), (1, f'{orig.max():.1f}')]:
            ax.text(i, frac, label, ha='center', va='center',
                    fontsize=7.5, color='#555555',
                    bbox=dict(fc='white', ec='none', pad=1))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(axes_cols, fontsize=10, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(-0.15, n_axes - 0.6)
    ax.set_ylim(-0.05, 1.05)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    legend_elems = [mpatches.Patch(color=REGION_COLORS[r], label=r) for r in ['东部', '中部', '西部', '东北']]
    ax.legend(handles=legend_elems, loc='lower right', fontsize=8, framealpha=0.85, ncol=2)

    ax.set_title('2024年各省多维经济指标平行坐标图\n'
                 'GDP总量 / GDP增速（2015-2024）/ 第三产业占比 / 人均GDP，粗线=前5大省',
                 fontsize=13, fontweight='bold', color='#222222')

    out = os.path.join(OUTPUT_DIR, '图表4_平行坐标图.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'已保存：{out}')


# ============================================================
# 图表5：桑基图（用 matplotlib 手绘流）
# ============================================================
def chart5_sankey(df_prov):
    """
    三层：地区 → 前10省+其余 → 三产业
    宽度 = 2024年GDP增量（2024-2015）
    """
    df_2015 = df_prov[df_prov['年份'] == 2015].set_index('省份')
    df_2024 = df_prov[df_prov['年份'] == 2024].set_index('省份')

    provinces = df_2024.index.tolist()
    inc = {}
    ind1 = {}; ind2 = {}; ind3 = {}
    for p in provinces:
        if p in df_2015.index:
            inc[p]  = df_2024.loc[p, '地区生产总值 (GDP)_亿元'] - df_2015.loc[p, '地区生产总值 (GDP)_亿元']
            ind1[p] = df_2024.loc[p, '第一产业增加值_亿元']
            ind2[p] = df_2024.loc[p, '第二产业增加值_亿元']
            ind3[p] = df_2024.loc[p, '第三产业增加值_亿元']

    # 按地区汇总
    region_inc = {'东部': 0, '中部': 0, '西部': 0, '东北': 0}
    for p, v in inc.items():
        region_inc[get_region(p)] += v

    total_inc = sum(region_inc.values())

    # 三产业汇总（2024全国）
    nat_ind1 = sum(ind1.values())
    nat_ind2 = sum(ind2.values())
    nat_ind3 = sum(ind3.values())
    nat_total = nat_ind1 + nat_ind2 + nat_ind3

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.axis('off')

    # ---- 手绘桑基流 ----
    # 列X坐标
    x0, x1, x2 = 0.05, 0.45, 0.85
    bar_w = 0.06

    def bezier_patch(ax, x_left, y_left_bot, h_left,
                     x_right, y_right_bot, h_right, color, alpha=0.55):
        """绘制一条从左矩形到右矩形的贝塞尔流带"""
        xl, xr = x_left + bar_w, x_right
        xm = (xl + xr) / 2
        verts = []
        # 上边缘
        t = np.linspace(0, 1, 60)
        top_y = (1 - t)**3 * (y_left_bot + h_left) \
              + 3*(1-t)**2*t * (y_left_bot + h_left) \
              + 3*(1-t)*t**2 * (y_right_bot + h_right) \
              + t**3         * (y_right_bot + h_right)
        top_x = (1-t)*xl + t*xr
        # 下边缘
        bot_y = (1 - t)**3 * y_left_bot \
              + 3*(1-t)**2*t * y_left_bot \
              + 3*(1-t)*t**2 * y_right_bot \
              + t**3         * y_right_bot
        bot_x = top_x
        xs = np.concatenate([top_x, bot_x[::-1]])
        ys = np.concatenate([top_y, bot_y[::-1]])
        ax.fill(xs, ys, color=color, alpha=alpha, lw=0)

    # ---- 第一列：地区 ----
    region_order = ['东部', '中部', '西部', '东北']
    region_y = {}
    cur = 0.05
    gap = 0.03
    for region in region_order:
        h = region_inc[region] / total_inc * 0.85
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, cur), bar_w, h,
            boxstyle='square,pad=0', fc=REGION_COLORS[region], ec='none', alpha=0.9))
        pct = region_inc[region] / total_inc * 100
        ax.text(x0 - 0.01, cur + h/2, f'{region}\n{pct:.1f}%',
                ha='right', va='center', fontsize=9.5, fontweight='bold',
                color=REGION_COLORS[region])
        region_y[region] = (cur, h)
        cur += h + gap

    # ---- 第二列：前8省 + 其余 ----
    inc_sorted = sorted(inc.items(), key=lambda x: x[1], reverse=True)
    top_n = 8
    top_provs = [x[0] for x in inc_sorted[:top_n]]
    other_inc  = sum(x[1] for x in inc_sorted[top_n:])
    prov_items = list(inc_sorted[:top_n]) + [('其他省份', other_inc)]

    prov_y = {}
    cur = 0.05
    gap2 = 0.015
    for prov, val in prov_items:
        h = val / total_inc * 0.85
        region = get_region(prov) if prov != '其他省份' else '东部'
        color  = REGION_COLORS[region]
        ax.add_patch(mpatches.FancyBboxPatch(
            (x1, cur), bar_w, h,
            boxstyle='square,pad=0', fc=color, ec='none', alpha=0.85))
        if h > 0.015:
            ax.text(x1 + bar_w/2, cur + h/2, prov,
                    ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')
        prov_y[prov] = (cur, h, region)
        cur += h + gap2

    # ---- 第三列：三产业 ----
    industry_colors = {'第一产业': '#2ca02c', '第二产业': '#1f77b4', '第三产业': '#ff7f0e'}
    ind_data = [
        ('第一产业', nat_ind1),
        ('第二产业', nat_ind2),
        ('第三产业', nat_ind3),
    ]
    ind_y = {}
    cur = 0.05
    gap3 = 0.04
    for ind_name, val in ind_data:
        h = val / nat_total * 0.85
        color = industry_colors[ind_name]
        ax.add_patch(mpatches.FancyBboxPatch(
            (x2, cur), bar_w, h,
            boxstyle='square,pad=0', fc=color, ec='none', alpha=0.9))
        ax.text(x2 + bar_w + 0.01, cur + h/2,
                f'{ind_name}\n{val/nat_total*100:.1f}%',
                ha='left', va='center', fontsize=9.5, fontweight='bold', color=color)
        ind_y[ind_name] = (cur, h)
        cur += h + gap3

    # ---- 流带：地区 → 省份 ----
    # 追踪每个地区在第一列的消费高度
    reg_flow_top = {r: region_y[r][0] + region_y[r][1] for r in region_order}

    for prov, val, reg in [(p, v, r) for p, (_, __, r) in prov_y.items()
                            for p2, v in inc.items() if p2 == p
                            for _ in [None]]:
        pass  # 简化：直接用 prov_y 中的信息

    # 重新组织：按地区分批画流
    for region in region_order:
        ry_bot, ry_h = region_y[region]
        region_total = region_inc[region]
        consume = 0
        for prov, (py_bot, py_h, preg) in prov_y.items():
            if preg != region:
                continue
            val = inc.get(prov, other_inc if prov == '其他省份' else 0)
            frac = val / region_total
            flow_h_left  = ry_h * frac
            y_left_bot   = ry_bot + consume
            bezier_patch(ax, x0, y_left_bot, flow_h_left,
                         x1, py_bot, py_h,
                         REGION_COLORS[region], alpha=0.30)
            consume += flow_h_left

    # ---- 流带：省份 → 产业（按各省2024产业比例） ----
    # 省份→产业分配使用全国平均比例（简化，数据中无省份增量产业拆分）
    for prov, (py_bot, py_h, preg) in prov_y.items():
        total_prov_gdp = sum(ind1.get(prov, 0) + ind2.get(prov, 0) + ind3.get(prov, 0)
                             for _ in [None])
        gdp_sum = ind1.get(prov, 0) + ind2.get(prov, 0) + ind3.get(prov, 0)
        if gdp_sum == 0:
            gdp_sum = nat_total
            ratios = [nat_ind1/nat_total, nat_ind2/nat_total, nat_ind3/nat_total]
        else:
            ratios = [ind1.get(prov, 0)/gdp_sum,
                      ind2.get(prov, 0)/gdp_sum,
                      ind3.get(prov, 0)/gdp_sum]
        consume = 0
        for (ind_name, _), ratio in zip(ind_data, ratios):
            iy_bot, iy_h = ind_y[ind_name]
            flow_h = py_h * ratio
            bezier_patch(ax, x1, py_bot + consume, flow_h,
                         x2, iy_bot + iy_h * (1 - ratio) * 0.5, flow_h,
                         industry_colors[ind_name], alpha=0.20)
            consume += flow_h

    # 列标题
    for x, label in [(x0 + bar_w/2, '地区'),
                     (x1 + bar_w/2, '主要省份'),
                     (x2 + bar_w/2, '产业')]:
        ax.text(x, 0.97, label, ha='center', va='bottom', fontsize=11,
                fontweight='bold', color='#333333')

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.02)
    ax.set_title('2024年全国GDP构成桑基图\n'
                 '地区 → 主要省份 → 三产业，宽度反映GDP总量占比',
                 fontsize=13, fontweight='bold', color='#222222', y=1.01)

    out = os.path.join(OUTPUT_DIR, '图表5_桑基图.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'已保存：{out}')


# ============================================================
# 图表6：PCA 双标图
# ============================================================
def chart6_pca(df_prov, df_detail):
    df_2015 = df_prov[df_prov['年份'] == 2015].set_index('省份')
    df_2024 = df_prov[df_prov['年份'] == 2024].set_index('省份')
    percap  = df_detail[df_detail['年份'] == 2024].set_index('地区')['人均地区生产总值']

    provinces = df_2024.index.tolist()
    features  = pd.DataFrame(index=provinces)
    features['GDP总量']       = df_2024['地区生产总值 (GDP)_亿元']
    features['GDP增速']       = (df_2024['地区生产总值 (GDP)_亿元'] - df_2015['地区生产总值 (GDP)_亿元']) \
                                 / df_2015['地区生产总值 (GDP)_亿元'] * 100
    features['第一产业占比']  = df_2024['第一产业占 GDP 比重_%']
    features['第二产业占比']  = df_2024['第二产业占 GDP 比重_%']
    features['第三产业占比']  = df_2024['第三产业占 GDP 比重_%']
    features['人均GDP']       = percap.reindex(provinces)
    features = features.dropna()

    feature_names = features.columns.tolist()
    X = StandardScaler().fit_transform(features.values)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    loadings = pca.components_.T  # shape (n_features, 2)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # 散点：省份
    for prov, (px, py) in zip(features.index, coords):
        region = get_region(prov)
        color  = REGION_COLORS[region]
        ax.scatter(px, py, color=color, s=70, alpha=0.85,
                   edgecolors='white', linewidths=0.8, zorder=4)
        ax.text(px + 0.08, py, prov, fontsize=7.5, color=color,
                va='center', alpha=0.9)

    # 箭头：特征载荷（放大3倍便于观察）
    scale = 3.5
    for i, fname in enumerate(feature_names):
        lx, ly = loadings[i, 0] * scale, loadings[i, 1] * scale
        ax.annotate('', xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.8))
        # 标注偏移，避免压字
        offset_x = 0.12 if lx >= 0 else -0.12
        offset_y = 0.10 if ly >= 0 else -0.12
        ax.text(lx + offset_x, ly + offset_y, fname,
                fontsize=9, color='#d62728', fontweight='bold', ha='center')

    ax.axhline(0, color='#cccccc', lw=0.8, ls='--')
    ax.axvline(0, color='#cccccc', lw=0.8, ls='--')

    ax.set_xlabel(f'PC1  ({var1:.1f}% 方差)', fontsize=11)
    ax.set_ylabel(f'PC2  ({var2:.1f}% 方差)', fontsize=11)
    ax.grid(alpha=0.25, ls='--')
    for spine in ax.spines.values():
        spine.set_edgecolor('#dddddd')

    legend_elems = [mpatches.Patch(color=REGION_COLORS[r], label=r) for r in ['东部', '中部', '西部', '东北']]
    ax.legend(handles=legend_elems, fontsize=8, framealpha=0.85, loc='lower left', ncol=2)

    ax.set_title(f'各省经济发展模式PCA双标图（2024年）\n'
                 f'散点=省份，红箭头=指标载荷方向，PC1+PC2解释{var1+var2:.1f}%方差',
                 fontsize=13, fontweight='bold', color='#222222')

    out = os.path.join(OUTPUT_DIR, '图表6_PCA双标图.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'已保存：{out}')


# ============================================================
# 图表7：3D 曲面图（省份 × 年份 × GDP）
# ============================================================
def chart7_3d_surface(df_prov):
    years    = sorted(df_prov['年份'].unique())
    # 按东部/中部/西部/东北排列省份
    order = EAST_CHINA + CENTRAL_CHINA + WEST_CHINA + NORTHEAST
    order = [p for p in order if p in df_prov['省份'].unique()]

    pivot = df_prov.pivot(index='年份', columns='省份', values='地区生产总值 (GDP)_亿元')
    pivot = pivot[order]

    X_idx = np.arange(len(years))
    Y_idx = np.arange(len(order))
    X, Y  = np.meshgrid(X_idx, Y_idx)
    Z     = pivot.values.T / 10000  # 万亿元

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(COLOR_BG)
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#f0f4f8')

    # 曲面颜色按Z值
    norm  = mcolors.Normalize(vmin=Z.min(), vmax=Z.max())
    cmap  = plt.cm.YlOrRd
    fcolors = cmap(norm(Z))

    surf = ax.plot_surface(X, Y, Z, facecolors=fcolors,
                           rstride=1, cstride=1,
                           linewidth=0.2, edgecolor='white',
                           alpha=0.9, antialiased=True)

    # 轴设置
    ax.set_xticks(X_idx)
    ax.set_xticklabels(years, fontsize=8, rotation=-15)
    ax.set_yticks(Y_idx[::3])
    ax.set_yticklabels(order[::3], fontsize=8)
    ax.set_xlabel('年份', fontsize=10, labelpad=8)
    ax.set_ylabel('省份（东部→中部→西部→东北）', fontsize=10, labelpad=12)
    ax.set_zlabel('GDP（万亿元）', fontsize=10, labelpad=8)

    # 颜色条
    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array([])
    cbar = fig.colorbar(m, ax=ax, shrink=0.4, aspect=15, pad=0.08)
    cbar.set_label('GDP（万亿元）', fontsize=9)

    ax.view_init(elev=30, azim=-60)

    ax.set_title('各省GDP规模时序地形图（2015-2024）\n'
                 '曲面高度=GDP总量，颜色越深越高，东部"山脉"远高于西部"平原"',
                 fontsize=12, fontweight='bold', color='#222222', pad=15)

    out = os.path.join(OUTPUT_DIR, '图表7_3D曲面图.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'已保存：{out}')


# ============================================================
# 图表8：3D 散点图（人均GDP × 增速 × 第三产业占比）
# ============================================================
def chart8_3d_scatter(df_prov, df_detail):
    df_2015 = df_prov[df_prov['年份'] == 2015].set_index('省份')
    df_2024 = df_prov[df_prov['年份'] == 2024].set_index('省份')
    percap  = df_detail[df_detail['年份'] == 2024].set_index('地区')['人均地区生产总值']

    provinces = df_2024.index.tolist()
    data = []
    for p in provinces:
        if p not in df_2015.index or p not in percap.index:
            continue
        gdp_growth = (df_2024.loc[p, '地区生产总值 (GDP)_亿元'] - df_2015.loc[p, '地区生产总值 (GDP)_亿元']) \
                     / df_2015.loc[p, '地区生产总值 (GDP)_亿元'] * 100
        ter_share  = df_2024.loc[p, '第三产业占 GDP 比重_%']
        pc_gdp     = percap.loc[p] / 10000  # 万元
        gdp_total  = df_2024.loc[p, '地区生产总值 (GDP)_亿元']
        region     = get_region(p)
        data.append((p, pc_gdp, gdp_growth, ter_share, gdp_total, region))

    df_plot = pd.DataFrame(data,
        columns=['省份', '人均GDP(万元)', 'GDP增速(%)', '第三产业占比(%)', 'GDP总量', '地区'])

    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor(COLOR_BG)
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#f0f4f8')

    size_scale = 500
    # 按z值（第三产业占比）排序，先画高z值的点，后画会被遮挡的低z值点
    df_plot = df_plot.sort_values('第三产业占比(%)', ascending=True)
    
    for _, row in df_plot.iterrows():
        color = REGION_COLORS[row['地区']]
        size  = np.sqrt(row['GDP总量'] / df_plot['GDP总量'].max()) * size_scale
        ax.scatter(row['人均GDP(万元)'], row['GDP增速(%)'], row['第三产业占比(%)'],
                   c=color, s=size, alpha=0.75,
                   edgecolors='white', linewidths=0.6, depthshade=True)
        
        # 添加到底面的投影线，帮助定位
        x, y, z = row['人均GDP(万元)'], row['GDP增速(%)'], row['第三产业占比(%)']
        ax.plot([x, x], [y, y], [0, z], color=color, alpha=0.2, linewidth=0.5)
        
        # 标注省份：只标注最重要省份，使用黑色文字+白色背景框提高对比度
        if row['GDP总量'] > 40000:  # 提高阈值，只显示最大省份
            ax.text(row['人均GDP(万元)'], row['GDP增速(%)'], row['第三产业占比(%)'] + 2.5,
                    row['省份'], fontsize=8, color='#222222', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))

    ax.set_xlabel('人均GDP（万元）', fontsize=10, labelpad=10)
    ax.set_ylabel('GDP增速（%）', fontsize=10, labelpad=10)
    ax.set_zlabel('第三产业占比（%）', fontsize=10, labelpad=10)
    ax.tick_params(labelsize=8)

    legend_elems = [mpatches.Patch(color=REGION_COLORS[r], label=r) for r in ['东部', '中部', '西部', '东北']]
    ax.legend(handles=legend_elems, fontsize=8, loc='upper left', framealpha=0.85, ncol=2)

    ax.view_init(elev=25, azim=-45)  # 调整视角减少遮挡

    ax.set_title('各省经济三维散点图（2024年）\n'
                 'X=人均GDP，Y=十年增速，Z=第三产业占比，点大小∝GDP总量',
                 fontsize=12, fontweight='bold', color='#222222', pad=15)

    out = os.path.join(OUTPUT_DIR, '图表8_3D散点图.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'已保存：{out}')


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print('加载数据...')
    df_prov, df_detail = load_data()

    print('生成图表4：平行坐标图...')
    chart4_parallel(df_prov, df_detail)

    print('生成图表5：桑基图...')
    chart5_sankey(df_prov)

    print('生成图表6：PCA双标图...')
    chart6_pca(df_prov, df_detail)

    print('生成图表7：3D曲面图...')
    chart7_3d_surface(df_prov)

    print('生成图表8：3D散点图...')
    chart8_3d_scatter(df_prov, df_detail)

    print('\n全部完成！')