import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from pathlib import Path
import numpy as np

def load_and_clean_data(file_path):
    """データの読み込みとクリーニングを行う関数"""
    # CSVファイルを読み込む
    df = pd.read_csv(file_path, encoding='utf-8', thousands=',', skiprows=3)
    
    # 不要な行を削除
    df = df.dropna(how='all')
    df = df[df['通勤・通学'].notna()]
    df = df[df['性別'] != '不明']
    
    # 年齢列を定義
    age_columns = ['～14歳', '15～19歳', '20～24歳', '25～29歳', '30～34歳', 
                  '35～39歳', '40～44歳', '45～49歳', '50～54歳', '55～59歳',
                  '60～64歳', '65～69歳', '70歳～']
    
    # 数値変換
    for col in age_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df, age_columns

def create_commute_age_plot(df, age_columns, output_dir):
    """通勤者の年齢分布グラフを作成"""
    plt.figure(figsize=(15, 8))
    commute_data = (df[(df['通勤・通学'] == '通勤') & (df['地域'] == '近畿圏計')]
                   .groupby('性別')[age_columns].sum())
    
    ax = commute_data.T.plot(kind='bar', stacked=True)
    plt.title('年齢階層別の通勤者数（性別）')
    plt.xlabel('年齢階層')
    plt.ylabel('人数')
    
    # 数値ラベルの追加
    for c in ax.containers:
        ax.bar_label(c, fmt='%.0f', label_type='center')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'commute_by_age_gender.png', dpi=300)
    plt.close()

def create_region_heatmap(df, age_columns, output_dir):
    """主要地域の通勤者数ヒートマップを作成"""
    target_regions = [
        '大阪北東部',
        '大阪北西部',
        '京都市',
        '京都中部',
        '京都南部',
        '近畿圏計'
    ]
    
    plt.figure(figsize=(30, 10))
    
    # データ準備
    male_data = df[(df['通勤・通学'] == '通勤') & 
                   (df['地域'].isin(target_regions)) &
                   (df['性別'] == '男性')].pivot_table(
        values=age_columns,
        index='地域',
        aggfunc='sum'
    ).astype(int)
    
    female_data = df[(df['通勤・通学'] == '通勤') & 
                     (df['地域'].isin(target_regions)) &
                     (df['性別'] == '女性')].pivot_table(
        values=age_columns,
        index='地域',
        aggfunc='sum'
    ).astype(int)
    
    # 合計計算
    male_total = male_data.sum(axis=1)
    female_total = female_data.sum(axis=1)
    
    # データ結合と列の順序変更
    heatmap_data = pd.DataFrame()
    
    # 年齢層の順序を調整（～14歳を最初に）
    age_columns_reordered = ['～14歳'] + [col for col in age_columns if col != '～14歳']
    
    # 男性データを追加
    for age in age_columns_reordered:
        heatmap_data[('男性', age)] = male_data[age]
    heatmap_data[('男性', '合計')] = male_total
    
    # 女性データを追加
    for age in age_columns_reordered:
        heatmap_data[('女性', age)] = female_data[age]
    heatmap_data[('女性', '合計')] = female_total
    
    # カラースケールの調整
    data_values = heatmap_data.values.flatten()
    data_values = data_values[data_values != 0]  # 0を除外
    
    # パーセンタイルの範囲をより細かく設定
    vmin = np.percentile(data_values, 1)    # 下位1%
    vmax = np.percentile(data_values, 99)   # 上位99%
    
    # ヒートマップの作成
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt=',d',
                cmap='YlOrRd',              # 黄色→オレンジ→赤のグラデーション
                cbar_kws={'label': '通勤者数'},
                xticklabels=True,
                square=False,
                annot_kws={'size': 8},
                vmin=vmin,
                vmax=vmax,
                center=None,                 # centerを削除して単調な色変化に
                robust=True)
    
    plt.title('地域別の性別・年齢層別通勤者数', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'commute_heatmap_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_student_age_plot(df, age_columns, output_dir):
    """通学者の年齢分布グラフを作成"""
    plt.figure(figsize=(15, 8))
    student_data = (df[(df['通勤・通学'] == '通学') & (df['地域'] == '近畿圏計')]
                   .groupby('性別')[age_columns].sum())
    
    ax = student_data.T.plot(kind='line', marker='o')
    
    plt.title('年齢階層別の通学者数（性別）')
    plt.xlabel('年齢階層')
    plt.ylabel('人数')
    plt.xticks(rotation=45)
    plt.legend(title='性別')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'student_by_age_gender.png', dpi=300)
    plt.close()

def main():
    # 出力ディレクトリの設定
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # データの読み込みとクリーニング
    df, age_columns = load_and_clean_data('001470675-a.csv')
    
    # グラフの生成
    create_commute_age_plot(df, age_columns, output_dir)
    create_region_heatmap(df, age_columns, output_dir)
    create_student_age_plot(df, age_columns, output_dir)
    
    print('グラフを生成しました。output/ディレクトリ内の以下のファイルを確認してください：')
    print('- commute_by_age_gender.png')
    print('- commute_heatmap_detailed.png')
    print('- student_by_age_gender.png')

if __name__ == '__main__':
    main() 