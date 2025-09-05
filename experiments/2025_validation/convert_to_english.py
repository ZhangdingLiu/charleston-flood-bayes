#!/usr/bin/env python3
"""
Convert Chinese text in aggressive_best_2017_09_11_threshold_0.3_experiment.json to English
"""

import json
import re
from datetime import datetime

def convert_to_english():
    # Load the original file
    with open('aggressive_best_2017_09_11_threshold_0.3_experiment.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Translation mapping
    translations = {
        # Strategy names
        "激进优化策略(增强覆盖率)": "Aggressive Optimization Strategy (Enhanced Coverage)",
        "激进优化": "Aggressive Optimization",
        
        # Network/method descriptions
        "enhanced_bayesian_network": "enhanced_bayesian_network",
        "full_network_enhanced_coverage": "full_network_enhanced_coverage",
        "enhanced_coverage_bayesian_inference": "enhanced_coverage_bayesian_inference",
        
        # Performance comparisons
        "vs 40 (基线)": "vs 40 (baseline)",
        "vs 39 (基线)": "vs 39 (baseline)", 
        "vs 50% (基线)": "vs 50% (baseline)",
        
        # Improvements
        "网络规模扩大至121个节点": "Network scale expanded to 121 nodes",
        "覆盖率提升至67.3% (35/52条道路)": "Coverage rate improved to 67.3% (35/52 roads)",
        "可预测111条道路状态": "Can predict status of 111 roads",
        "使用全历史数据训练(855条记录)": "Trained using full historical data (855 records)",
        
        # Improvement metrics
        "+17.3个百分点": "+17.3 percentage points",
        "+81个节点": "+81 nodes", 
        "+78条道路": "+78 roads",
        
        # Advantages
        "覆盖率大幅提升(67.3%)": "Significant coverage improvement (67.3%)",
        "网络规模显著扩大(121节点)": "Substantial network expansion (121 nodes)",
        "可监控更多道路": "Can monitor more roads",
        "精度依然保持100%": "Precision maintained at 100%",
        
        # Challenges
        "召回率相对较低(16.0%)": "Relatively low recall rate (16.0%)",
        "大网络导致预测概率普遍偏低": "Large network leads to generally low prediction probabilities",
        "稀疏数据问题": "Sparse data problem",
        "F1分数(0.276)低于基线": "F1 score (0.276) lower than baseline",
        
        # Recommendations
        "适用于最大化监控覆盖范围的场景": "Suitable for scenarios requiring maximum monitoring coverage",
        "可容忍较多漏报但要求零误报": "Can tolerate more false negatives but requires zero false positives",
        "建议结合多个阈值进行预警分级": "Recommend combining multiple thresholds for alert stratification",
        "考虑参数微调平衡覆盖率与性能": "Consider parameter tuning to balance coverage and performance",
        
        # Data descriptions
        "全历史数据(除测试日期)": "Full historical data (excluding test date)",
        "2017年9月11日洪水事件 - Charleston最严重洪水之一": "September 11, 2017 flood event - One of Charleston's most severe floods"
    }
    
    # Convert the entire data structure to string, replace, then convert back
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    
    # Apply translations
    for chinese, english in translations.items():
        json_str = json_str.replace(chinese, english)
    
    # Convert back to dict
    data_english = json.loads(json_str)
    
    # Update metadata
    data_english["metadata"]["generation_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data_english["metadata"]["language"] = "English"
    data_english["metadata"]["converted_from"] = "aggressive_best_2017_09_11_threshold_0.3_experiment.json"
    
    # Save English version
    output_file = "aggressive_best_2017_09_11_threshold_0.3_experiment_english.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_english, f, indent=2, ensure_ascii=False)
    
    print(f"✅ English version saved as: {output_file}")
    
    # Show some key translations
    print("\n📝 Key translations applied:")
    print("=" * 50)
    for chinese, english in list(translations.items())[:10]:
        print(f"• {chinese[:30]}... → {english[:30]}...")
    
    return output_file

if __name__ == "__main__":
    convert_to_english()