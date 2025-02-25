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
        '始発～6:59', '7:00～7:29', '7:30～7:59', '8:00～8:29', '8:30～8:59',
        '9:00～9:29', '9:30～9:59', '10:00～10:59', '11:00～12:59', '13:00～14:59',
        '15:00～16:59', '17:00～17:59', '18:00～18:59', '19:00～19:59', '20:00～20:59',
        '21:00～21:59', '22:00～22:59', '23:00～23:59', '24:00～終発'
    ]
    df.columns = new_columns
    
    # 空白を前の有効な値で埋める（事業者名、路線、方向）
    fill_columns = ['事業者名', '路線', '方向']
    for col in fill_columns:
        df[col] = df[col].fillna(method='ffill')
        
    # 発駅と着駅が両方空白の行を削除
    df = df.dropna(subset=['発駅', '着駅'], how='all')
    
    # 数値データのカンマを除去して数値に変換
    for col in df.columns[5:]:  # 時間帯の列のみを処理
        df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
    
    # データを長形式（ロング形式）に変換
    df_melted = df.melt(
        id_vars=['事業者名', '路線', '方向', '発駅', '着駅'],
        var_name='時間帯',
        value_name='輸送人員'
    )
    
    # 時間帯の順序を定義
    time_order = [
        '始発-6時台', '7時前半', '7時後半', '8時前半', '8時後半',
        '9時前半', '9時後半', '10時台', '11-12時台', '13-14時台',
        '15-16時台', '17時台', '18時台', '19時台', '20時台',
        '21時台', '22時台', '23時台', '24時以降'
    ]
    
    # 時間帯を文字列にマッピング
    time_mapping = {
        '始発～6:59': '始発-6時台',
        '7:00～7:29': '7時前半',
        '7:30～7:59': '7時後半',
        '8:00～8:29': '8時前半',
        '8:30～8:59': '8時後半',
        '9:00～9:29': '9時前半',
        '9:30～9:59': '9時後半',
        '10:00～10:59': '10時台',
        '11:00～12:59': '11-12時台',
        '13:00～14:59': '13-14時台',
        '15:00～16:59': '15-16時台',
        '17:00～17:59': '17時台',
        '18:00～18:59': '18時台',
        '19:00～19:59': '19時台',
        '20:00～20:59': '20時台',
        '21:00～21:59': '21時台',
        '22:00～22:59': '22時台',
        '23:00～23:59': '23時台',
        '24:00～終発': '24時以降'
    }
    df_melted['時間帯'] = df_melted['時間帯'].map(time_mapping)
    
    # 時間帯を順序付きカテゴリ型に変換
    df_melted['時間帯'] = pd.Categorical(df_melted['時間帯'], categories=time_order, ordered=True)
    
    # 発駅と着駅の空白を処理
    df_melted['発駅'] = df_melted['発駅'].fillna('')
    df_melted['着駅'] = df_melted['着駅'].fillna('')
    df_melted['駅間'] = df_melted['発駅'] + '-' + df_melted['着駅']
    
    # 輸送人員が欠損している行のみを削除
    df_melted = df_melted.dropna(subset=['輸送人員'])
    
    return df_melted

# ヒートマップの作成
def create_heatmap(df, line_name, direction, start_company='京阪電気鉄道'):
    # 時間帯の順序を定義
    time_order = [
        '始発-6時台', '7時前半', '7時後半', '8時前半', '8時後半',
        '9時前半', '9時後半', '10時台', '11-12時台', '13-14時台',
        '15-16時台', '17時台', '18時台', '19時台', '20時台',
        '21時台', '22時台', '23時台', '24時以降'
    ]
    
    # 京阪電気鉄道以降のデータを抽出
    start_idx = df[df['事業者名'] == start_company].index[0]
    filtered_df = df.iloc[start_idx:]
    
    # 特定の路線と方向のデータを抽出
    line_data = filtered_df[
        (filtered_df['路線'] == line_name) & 
        (filtered_df['方向'] == direction)
    ]
    
    # 駅間と時間帯のピボットテーブル作成
    pivot_data = pd.pivot_table(
        line_data,
        values='輸送人員',
        index='発駅',
        columns='時間帯',
        aggfunc='mean'
    )
    
    # 駅の順序を取得（データフレームの順序を使用）
    station_order = line_data['発駅'].unique()
    
    # ピボットテーブルを路線順に並び替え
    pivot_data = pivot_data.reindex(index=station_order, columns=time_order)
    
    # データが空の場合は処理を中断
    if pivot_data.empty:
        print(f"警告: {line_name}（{direction}）のデータが見つかりません")
        return
    
    # プロットサイズの設定
    plt.figure(figsize=(15, len(pivot_data)/2))
    
    # ヒートマップの作成
    sns.heatmap(
        pivot_data,
        cmap='YlOrRd',
        fmt='.0f',
        cbar_kws={'label': '輸送人員（人）'}
    )
    
    # x軸のラベルを調整
    plt.xticks(rotation=45, ha='right')
    
    plt.title(f'{line_name}（{direction}）時間帯別輸送人員ヒートマップ')
    plt.xlabel('時間帯')
    plt.ylabel('発駅')
    plt.tight_layout()
    plt.savefig(f'heatmap_{line_name}_{direction}.png')
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
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('時間帯')
    plt.ylabel('平均輸送人員（人）')
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
    
    # 京阪本線の上り下りそれぞれのヒートマップを作成
    line_name = '京阪本線'
    for direction in ['下り', '上り']:
        create_heatmap(df, line_name, direction, start_company='京阪電気鉄道')
    
    # その他の処理は同じ
    create_station_timeline(df, '京都')
    create_line_comparison(df)
    df.to_excel('analyzed_data.xlsx', index=False)
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