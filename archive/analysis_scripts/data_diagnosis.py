import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
import itertools
from typing import Dict, List, Tuple, Set
import json

class FloodDataDiagnosis:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data = None
        self.flood_data = None
        self.street_counts = None
        self.daily_cooccurrence = None
        
    def load_and_filter_data(self) -> pd.DataFrame:
        """加载Road_Closures_2024.csv并筛选REASON为FLOOD的记录"""
        print("Loading data...")
        self.data = pd.read_csv(self.csv_path)
        
        print(f"总记录数: {len(self.data)}")
        print(f"列名: {list(self.data.columns)}")
        
        # 筛选洪水记录
        self.flood_data = self.data[self.data['REASON'] == 'FLOOD'].copy()
        print(f"洪水记录数: {len(self.flood_data)}")
        
        # 转换时间格式
        if 'START' in self.flood_data.columns:
            self.flood_data['START'] = pd.to_datetime(self.flood_data['START'], errors='coerce')
            self.flood_data['date'] = self.flood_data['START'].dt.date
        
        return self.flood_data
    
    def analyze_street_frequency(self) -> Dict[str, int]:
        """统计每条道路的洪水出现频次"""
        print("\n=== 道路洪水频次分析 ===")
        
        self.street_counts = Counter(self.flood_data['STREET'].dropna())
        
        print(f"受影响道路总数: {len(self.street_counts)}")
        print(f"洪水事件总数: {sum(self.street_counts.values())}")
        
        # 频次分布统计
        frequency_dist = Counter(self.street_counts.values())
        print("\n频次分布:")
        for freq, count in sorted(frequency_dist.items()):
            print(f"  出现{freq}次的道路: {count}条")
        
        # 显示最频繁的道路
        print("\n最频繁洪水道路 (前20):")
        for street, count in self.street_counts.most_common(20):
            print(f"  {street}: {count}次")
        
        return self.street_counts
    
    def analyze_daily_cooccurrence(self) -> Dict[str, Set[str]]:
        """分析按天分组后道路之间的共现模式"""
        print("\n=== 道路共现模式分析 ===")
        
        # 按日期分组
        daily_streets = defaultdict(set)
        for _, row in self.flood_data.iterrows():
            if pd.notna(row['date']) and pd.notna(row['STREET']):
                daily_streets[str(row['date'])].add(row['STREET'])
        
        print(f"有洪水记录的天数: {len(daily_streets)}")
        
        # 计算共现矩阵
        cooccurrence_matrix = defaultdict(lambda: defaultdict(int))
        cooccurrence_pairs = Counter()
        
        for date, streets in daily_streets.items():
            street_list = list(streets)
            if len(street_list) > 1:
                # 计算该天所有道路对的共现
                for street1, street2 in itertools.combinations(street_list, 2):
                    cooccurrence_matrix[street1][street2] += 1
                    cooccurrence_matrix[street2][street1] += 1
                    pair = tuple(sorted([street1, street2]))
                    cooccurrence_pairs[pair] += 1
        
        print(f"共现道路对数量: {len(cooccurrence_pairs)}")
        
        # 显示每日道路数量分布
        daily_counts = [len(streets) for streets in daily_streets.values()]
        print(f"\n每日平均道路数: {np.mean(daily_counts):.2f}")
        print(f"每日最多道路数: {max(daily_counts)}")
        print(f"每日最少道路数: {min(daily_counts)}")
        
        # 显示最频繁的共现对
        print("\n最频繁共现道路对 (前15):")
        for pair, count in cooccurrence_pairs.most_common(15):
            print(f"  {pair[0]} - {pair[1]}: {count}次")
        
        self.daily_cooccurrence = cooccurrence_matrix
        return cooccurrence_matrix
    
    def analyze_parameter_thresholds(self) -> Dict[str, Dict]:
        """计算不同参数阈值下会保留多少道路和连接"""
        print("\n=== 参数阈值分析 ===")
        
        # 不同频次阈值的道路保留情况
        frequency_thresholds = [1, 2, 3, 5, 10, 15, 20, 30]
        threshold_analysis = {}
        
        for threshold in frequency_thresholds:
            retained_streets = [street for street, count in self.street_counts.items() 
                              if count >= threshold]
            
            # 计算保留的连接数
            retained_connections = 0
            for street1 in retained_streets:
                for street2 in retained_streets:
                    if street1 < street2 and street2 in self.daily_cooccurrence[street1]:
                        retained_connections += 1
            
            threshold_analysis[threshold] = {
                'retained_streets': len(retained_streets),
                'retained_connections': retained_connections,
                'street_percentage': len(retained_streets) / len(self.street_counts) * 100,
                'streets': retained_streets
            }
            
            print(f"频次阈值 >= {threshold}: "
                  f"保留道路 {len(retained_streets)}/{len(self.street_counts)} "
                  f"({len(retained_streets)/len(self.street_counts)*100:.1f}%), "
                  f"连接数 {retained_connections}")
        
        # 不同共现阈值的连接保留情况
        cooccurrence_thresholds = [1, 2, 3, 5, 10]
        cooccurrence_analysis = {}
        
        for threshold in cooccurrence_thresholds:
            retained_pairs = 0
            for street1 in self.daily_cooccurrence:
                for street2, count in self.daily_cooccurrence[street1].items():
                    if street1 < street2 and count >= threshold:
                        retained_pairs += 1
            
            cooccurrence_analysis[threshold] = {
                'retained_pairs': retained_pairs,
                'threshold': threshold
            }
            
            print(f"共现阈值 >= {threshold}: 保留连接对 {retained_pairs}")
        
        return {
            'frequency_analysis': threshold_analysis,
            'cooccurrence_analysis': cooccurrence_analysis
        }
    
    def generate_detailed_report(self) -> str:
        """生成详细的统计报告和参数建议"""
        report = []
        report.append("=" * 60)
        report.append("Charleston洪水数据质量和特征分析报告")
        report.append("=" * 60)
        
        # 基本统计信息
        report.append(f"\n【基本统计信息】")
        report.append(f"- 总记录数: {len(self.data)}")
        report.append(f"- 洪水记录数: {len(self.flood_data)}")
        report.append(f"- 受影响道路总数: {len(self.street_counts)}")
        report.append(f"- 洪水事件总数: {sum(self.street_counts.values())}")
        
        # 时间范围分析
        if 'START' in self.flood_data.columns:
            start_dates = self.flood_data['START'].dropna()
            if len(start_dates) > 0:
                report.append(f"- 时间范围: {start_dates.min()} 至 {start_dates.max()}")
        
        # 数据稀疏性分析
        report.append(f"\n【数据稀疏性分析】")
        
        # 频次分布
        frequency_dist = Counter(self.street_counts.values())
        report.append(f"- 只出现1次的道路: {frequency_dist.get(1, 0)}条 ({frequency_dist.get(1, 0)/len(self.street_counts)*100:.1f}%)")
        report.append(f"- 出现2-5次的道路: {sum(frequency_dist.get(i, 0) for i in range(2, 6))}条")
        report.append(f"- 出现6-10次的道路: {sum(frequency_dist.get(i, 0) for i in range(6, 11))}条")
        report.append(f"- 出现10次以上的道路: {sum(frequency_dist.get(i, 0) for i in range(11, max(frequency_dist.keys())+1))}条")
        
        # 共现分析
        total_possible_pairs = len(self.street_counts) * (len(self.street_counts) - 1) // 2
        actual_pairs = sum(len(neighbors) for neighbors in self.daily_cooccurrence.values()) // 2
        report.append(f"- 可能的道路对数: {total_possible_pairs}")
        report.append(f"- 实际共现对数: {actual_pairs}")
        report.append(f"- 连接密度: {actual_pairs/total_possible_pairs*100:.2f}%")
        
        # 参数建议
        report.append(f"\n【贝叶斯网络参数建议】")
        
        # 基于数据稀疏性的建议
        if frequency_dist.get(1, 0) / len(self.street_counts) > 0.5:
            report.append("⚠️  数据稀疏性警告: 超过50%的道路只出现1次")
            report.append("建议: 考虑提高频次阈值或合并相似道路")
        
        # 频次阈值建议
        for threshold in [2, 3, 5, 10]:
            retained = sum(1 for count in self.street_counts.values() if count >= threshold)
            percentage = retained / len(self.street_counts) * 100
            if 20 <= percentage <= 80:
                report.append(f"推荐频次阈值: {threshold} (保留{retained}条道路, {percentage:.1f}%)")
                break
        
        # 网络复杂度建议
        if actual_pairs > 1000:
            report.append("⚠️  网络复杂度警告: 连接数过多可能影响计算效率")
            report.append("建议: 增加共现阈值或使用分层建模")
        
        # 数据质量建议
        missing_dates = self.flood_data['START'].isna().sum()
        if missing_dates > 0:
            report.append(f"⚠️  数据质量问题: {missing_dates}条记录缺少时间信息")
        
        return "\n".join(report)
    
    def save_analysis_results(self, output_path: str = "flood_analysis_results.json"):
        """保存分析结果到JSON文件"""
        results = {
            'basic_stats': {
                'total_records': len(self.data),
                'flood_records': len(self.flood_data),
                'affected_streets': len(self.street_counts),
                'total_events': sum(self.street_counts.values())
            },
            'street_frequency': dict(self.street_counts),
            'cooccurrence_matrix': {
                street1: dict(neighbors) 
                for street1, neighbors in self.daily_cooccurrence.items()
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"分析结果已保存到: {output_path}")
    
    def run_full_analysis(self):
        """运行完整的数据诊断分析"""
        print("开始Charleston洪水数据诊断分析...")
        
        # 步骤1: 加载和筛选数据
        self.load_and_filter_data()
        
        # 步骤2: 道路频次分析
        self.analyze_street_frequency()
        
        # 步骤3: 共现模式分析
        self.analyze_daily_cooccurrence()
        
        # 步骤4: 参数阈值分析
        self.analyze_parameter_thresholds()
        
        # 步骤5: 生成详细报告
        report = self.generate_detailed_report()
        print("\n" + report)
        
        # 步骤6: 保存结果
        self.save_analysis_results()
        
        return report

def main():
    """主函数"""
    csv_path = "Road_Closures_2024.csv"
    
    try:
        diagnosis = FloodDataDiagnosis(csv_path)
        diagnosis.run_full_analysis()
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_path}")
        print("请确保文件在当前目录中")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")

if __name__ == "__main__":
    main()