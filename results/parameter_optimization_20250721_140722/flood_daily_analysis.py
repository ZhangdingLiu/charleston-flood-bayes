#!/usr/bin/env python3
"""
快速分析每日flood道路数量，找出Top10洪水日期
"""

import sys
import os
sys.path.append('/mnt/d/Data/coda_PycharmProjects/PIN_bayesian')

import pandas as pd

def analyze_daily_flood_events():
    """分析每日洪水事件，返回Top10"""
    
    # 1. 加载数据
    df = pd.read_csv("/mnt/d/Data/coda_PycharmProjects/PIN_bayesian/Road_Closures_2024.csv")
    flood_df = df[df["REASON"].str.upper() == "FLOOD"].copy()
    
    # 2. 时间处理
    flood_df["time_create"] = pd.to_datetime(flood_df["START"], utc=True)
    flood_df["date"] = flood_df["time_create"].dt.date
    
    # 3. 每日统计
    daily_stats = flood_df.groupby("date").agg({
        'STREET': 'nunique',  # 独特道路数
        'OBJECTID': 'count'   # 总记录数
    }).rename(columns={'STREET': 'unique_roads', 'OBJECTID': 'total_records'})
    
    # 4. Top 10
    top10_dates = daily_stats.nlargest(10, 'unique_roads')
    
    print("📊 Top 10 洪水日期 (按道路数量排序):")
    print("=" * 60)
    
    for i, (date, row) in enumerate(top10_dates.iterrows(), 1):
        # 获取该日期的具体道路
        day_roads = flood_df[flood_df["date"] == date]["STREET"].unique()
        print(f"{i:2d}. {date} - {row['unique_roads']:2d}条道路, {row['total_records']:2d}条记录")
        if len(day_roads) <= 8:
            print(f"    道路: {', '.join(day_roads)}")
        else:
            print(f"    道路: {', '.join(day_roads[:8])}... (共{len(day_roads)}条)")
        print()
    
    # 5. 保存结果
    top10_dates.to_csv("top10_flood_dates.csv")
    
    # 6. 基本统计
    print("📈 基本统计:")
    print(f"   总洪水记录: {len(flood_df)}")
    print(f"   洪水日期数: {len(daily_stats)}")
    print(f"   平均每日道路数: {daily_stats['unique_roads'].mean():.1f}")
    print(f"   最大单日道路数: {daily_stats['unique_roads'].max()}")
    
    return top10_dates

if __name__ == "__main__":
    results = analyze_daily_flood_events()