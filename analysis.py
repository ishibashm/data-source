import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
from pathlib import Path

# データの読み込みと前処理
def load_and_preprocess_data(file_path):
    # ヘッダーを飛ばして2行目からデータを読み込む
    df = pd.read_csv(file_path, encoding='utf-8', skiprows=1)
    
    # 実際の列名を使用
    new_columns = [
        '事業者名', '路線', '方向', '発駅', '着駅',
        '6時以前', '7時前半', '7時後半', '8時前半', '8時後半',
        '9時前半', '9時後半', '10時台', '11-12時台', '13-14時台',
        '15-16時台', '17時台', '18時台', '19時台', '20時台',
        '21時台', '22時台', '23時台', '24時以降'
    ]
    df.columns = new_columns
    
    # 数値データのカンマを除去して数値に変換
    for col in df.columns[5:]:  # 時間帯の列のみを処理
        df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
    
    # データを長形式（ロング形式）に変換
    df_melted = df.melt(
        id_vars=['事業者名', '路線', '方向', '発駅', '着駅'],
        var_name='時間帯',
        value_name='輸送人員'
    )
    
    # 時間帯を数値に変換（簡略化のため、時間帯の中央値を使用）
    time_mapping = {
        '6時以前': 6, '7時前半': 7, '7時後半': 7.5, '8時前半': 8, '8時後半': 8.5,
        '9時前半': 9, '9時後半': 9.5, '10時台': 10, '11-12時台': 11.5, '13-14時台': 13.5,
        '15-16時台': 15.5, '17時台': 17, '18時台': 18, '19時台': 19, '20時台': 20,
        '21時台': 21, '22時台': 22, '23時台': 23, '24時以降': 24
    }
    df_melted['時間帯'] = df_melted['時間帯'].map(time_mapping)
    
    # 駅間を作成（NaNを空文字列に置換）
    df_melted['発駅'] = df_melted['発駅'].fillna('')
    df_melted['着駅'] = df_melted['着駅'].fillna('')
    df_melted['駅間'] = df_melted['発駅'] + '-' + df_melted['着駅']
    
    # NaNを含む行を削除
    df_melted = df_melted.dropna()
    
    return df_melted

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
    # 発駅または着駅に指定駅が含まれるデータを抽出
    station_data = df[df['発駅'].str.contains(station_name, na=False) | 
                     df['着駅'].str.contains(station_name, na=False)]
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
    df = load_and_preprocess_data('001179022.csv')
    
    # 各グラフの作成
    create_heatmap(df)
    create_station_timeline(df, '京都')
    create_line_comparison(df)
    
    # 整理したデータをExcelに出力
    df.to_excel('analyzed_data.xlsx', index=False)
    
    # 分析結果のレポート作成
    create_report(df)

def create_report(df):
    # 京都駅のデータを抽出
    kyoto_data = df[df['発駅'].str.contains('京都', na=False) | df['着駅'].str.contains('京都', na=False)]
    
    # 京都駅のピーク時間と最小時間を取得（データがある場合のみ）
    if len(kyoto_data) > 0:
        kyoto_peak = kyoto_data.groupby('時間帯')['輸送人員'].mean().idxmax()
        kyoto_min = kyoto_data.groupby('時間帯')['輸送人員'].mean().idxmin()
    else:
        kyoto_peak = "データなし"
        kyoto_min = "データなし"

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
  - ピーク時間帯：{kyoto_peak}
  - 最小時間帯：{kyoto_min}

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
        kyoto_peak=kyoto_peak,
        kyoto_min=kyoto_min,
        top_line=df.groupby('路線')['輸送人員'].mean().idxmax(),
        bottom_line=df.groupby('路線')['輸送人員'].mean().idxmin()
    )
    
    with open('analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

if __name__ == "__main__":
    main() 