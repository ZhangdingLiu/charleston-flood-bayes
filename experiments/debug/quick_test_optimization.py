#!/usr/bin/env python3
"""
Quick Test Parameter Optimization
快速测试参数优化系统

最简化的测试版本，确保系统正常工作
4个参数组合 + 基础可视化

作者：Claude AI
日期：2025-01-21
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'analysis'))
sys.path.append(os.path.join(current_dir, 'visualization'))

def quick_test():
    """快速测试参数优化系统"""
    print("⚡ 快速参数优化系统测试")
    print("=" * 50)
    print("测试配置: 4个参数组合")
    print("预计时间: 2-3分钟")
    print()
    
    try:
        from analysis.comprehensive_parameter_grid_search import ParameterGridSearcher
        from visualization.parameter_analysis_visualizer import ParameterVisualizer
        import pandas as pd
        
        print("✅ 模块导入成功")
        
        # 定义测试参数网格
        test_param_grid = {
            'occ_thr': [3],                    # 1个值
            'edge_thr': [2],                   # 1个值
            'weight_thr': [0.3, 0.4],          # 2个值
            'evidence_count': [2, 3],          # 2个值
            'pred_threshold': [0.2],           # 1个值
            'neg_pos_ratio': [1.0],            # 1个值
            'marginal_prob_threshold': [0.05]  # 1个值
        }
        
        print(f"📊 参数网格: {2*2} = 4个组合")
        
        # 运行网格搜索
        print("\n🔍 开始网格搜索...")
        searcher = ParameterGridSearcher(param_grid=test_param_grid)
        results, result_dir = searcher.run_grid_search(save_dir='results/quick_test')
        
        if not results:
            print("❌ 网格搜索失败")
            return False
        
        df_results = pd.DataFrame(results)
        print(f"✅ 网格搜索成功: {len(results)}/4 个组合")
        
        # 显示结果摘要
        print(f"\n📈 性能摘要:")
        print(f"F1分数范围: {df_results['f1_score'].min():.3f} - {df_results['f1_score'].max():.3f}")
        print(f"精确度范围: {df_results['precision'].min():.3f} - {df_results['precision'].max():.3f}")
        print(f"召回率范围: {df_results['recall'].min():.3f} - {df_results['recall'].max():.3f}")
        
        # 找到最佳配置
        best_idx = df_results['f1_score'].idxmax()
        best_config = df_results.iloc[best_idx]
        
        print(f"\n🏆 最佳配置 (F1={best_config['f1_score']:.3f}):")
        print(f"  参数: weight_thr={best_config['weight_thr']}, evidence_count={best_config['evidence_count']}")
        print(f"  性能: P={best_config['precision']:.3f}, R={best_config['recall']:.3f}")
        
        # 测试可视化
        print(f"\n🎨 生成可视化...")
        viz_dir = os.path.join(result_dir, 'visualizations')
        visualizer = ParameterVisualizer(df_results, viz_dir)
        
        # 生成3D图
        visualizer.create_3d_performance_scatter()
        
        # 测试约束条件
        constraints = {'min_precision': 0.6, 'min_recall': 0.3}
        filtered_df = visualizer.create_constraint_filtering_visualization(constraints)
        
        print(f"✅ 可视化生成成功")
        print(f"  - 3D性能分布图")
        print(f"  - 约束筛选图 (筛选出 {len(filtered_df)} 个配置)")
        
        # 保存简要报告
        report_file = os.path.join(result_dir, "quick_test_summary.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("快速参数优化测试结果\n")
            f.write("="*30 + "\n\n")
            f.write(f"测试时间: {pd.Timestamp.now()}\n")
            f.write(f"参数组合数: 4\n")
            f.write(f"成功组合数: {len(results)}\n\n")
            f.write("性能统计:\n")
            f.write(f"  最佳F1: {df_results['f1_score'].max():.3f}\n")
            f.write(f"  最高精确度: {df_results['precision'].max():.3f}\n")
            f.write(f"  最高召回率: {df_results['recall'].max():.3f}\n\n")
            f.write("推荐配置:\n")
            f.write(f"  weight_thr: {best_config['weight_thr']}\n")
            f.write(f"  evidence_count: {best_config['evidence_count']}\n")
        
        print(f"\n🎉 快速测试完全成功！")
        print(f"📁 结果目录: {result_dir}")
        print(f"📄 测试报告: {report_file}")
        print(f"📊 完整结果: {result_dir}/complete_results.csv")
        print(f"🎨 可视化: {viz_dir}/")
        
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        print("请确保analysis和visualization目录存在")
        return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("⚡ 贝叶斯网络参数优化系统 - 快速测试")
    print("=" * 60)
    print("本测试将验证参数优化系统是否正常工作")
    print("测试内容：4个参数组合 + 基础可视化")
    print("=" * 60)
    
    # 询问用户
    user_input = input("\n是否开始快速测试？(y/n): ").lower()
    if user_input not in ['y', 'yes', '是', '']:
        print("测试已取消")
        return
    
    # 运行测试
    success = quick_test()
    
    if success:
        print("\n" + "=" * 60)
        print("🎊 快速测试成功！系统可以正常使用")
        print("💡 接下来可以运行:")
        print("   - python test_parameter_optimization.py  (更多参数测试)")
        print("   - python run_parameter_optimization.py   (完整优化)")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 测试失败，请检查错误信息")
        print("=" * 60)

if __name__ == "__main__":
    main()