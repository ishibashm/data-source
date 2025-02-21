import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
from pathlib import Path

# データの読み込みと前処理
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    
    # 時間帯を数値形式に変換
    df['時間帯'] = df['時間帯'].str.extract('(\d+)').astype(int)
    
    # 駅間の文字列を統一（表記ゆれの修正）
    df['駅間'] = df['駅間'].str.strip()
    
    return df

# ヒートマップの作成
def create_heatmap(df):
    pivot_table = df.pivot_table(
        values='輸送人員',
        index='駅間',
        columns='時間帯',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='.0f')
    plt.title('時間帯別・駅間の輸送人員ヒートマップ')
    plt.xlabel('時間帯')
    plt.ylabel('駅間')
    plt.tight_layout()
    plt.savefig('heatmap.png')
    plt.close()

# 特定駅の時間帯別輸送人員の折れ線グラフ
def create_station_timeline(df, station_name):
    station_data = df[df['駅間'].str.contains(station_name)]
    timeline = station_data.groupby('時間帯')['輸送人員'].mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(timeline.index, timeline.values, marker='o')
    plt.title(f'{station_name}の時間帯別輸送人員')
    plt.xlabel('時間帯')
    plt.ylabel('平均輸送人員')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('station_timeline.png')
    plt.close()

# 路線別平均輸送人員の棒グラフ
def create_line_comparison(df):
    line_avg = df.groupby('路線')['輸送人員'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    line_avg.plot(kind='bar')
    plt.title('路線別平均輸送人員')
    plt.xlabel('路線')
    plt.ylabel('平均輸送人員')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('line_comparison.png')
    plt.close()

# メイン処理
def main():
    # データの読み込み
    df = load_and_preprocess_data('data.csv')
    
    # 各グラフの作成
    create_heatmap(df)
    create_station_timeline(df, '京都')
    create_line_comparison(df)
    
    # 整理したデータをExcelに出力
    df.to_excel('analyzed_data.xlsx', index=False)
    
    # 分析結果のレポート作成
    create_report(df)

def create_report(df):
    report = """# 鉄道輸送人員データ分析レポート

## 1. データ概要
- 総レコード数: {total_records}
- 対象路線数: {total_lines}
- 対象駅間数: {total_stations}

## 2. 分析結果

### 2.1 時間帯別・駅間の輸送人員分布
- ヒートマップ（heatmap.png）から、以下の特徴が観察されました：
  - 最も輸送人員が多い時間帯：{peak_hour}時台
  - 最も輸送人員が多い駅間：{busiest_section}

### 2.2 京都駅の時間帯別分析
- 時間推移グラフ（station_timeline.png）から：
  - ピーク時間帯：{kyoto_peak}時台
  - 最小時間帯：{kyoto_min}時台

### 2.3 路線別比較
- 棒グラフ（line_comparison.png）から：
  - 最も輸送人員が多い路線：{top_line}
  - 最も輸送人員が少ない路線：{bottom_line}
    """.format(
        total_records=len(df),
        total_lines=df['路線'].nunique(),
        total_stations=df['駅間'].nunique(),
        peak_hour=df.groupby('時間帯')['輸送人員'].mean().idxmax(),
        busiest_section=df.groupby('駅間')['輸送人員'].mean().idxmax(),
        kyoto_peak=df[df['駅間'].str.contains('京都')].groupby('時間帯')['輸送人員'].mean().idxmax(),
        kyoto_min=df[df['駅間'].str.contains('京都')].groupby('時間帯')['輸送人員'].mean().idxmin(),
        top_line=df.groupby('路線')['輸送人員'].mean().idxmax(),
        bottom_line=df.groupby('路線')['輸送人員'].mean().idxmin()
    )
    
    with open('analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

if __name__ == "__main__":
    main() 