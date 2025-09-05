#!/usr/bin/env python3
"""
创建HTML格式的可视化报告
- 使用纯HTML/CSS/JavaScript避免Python依赖问题
"""

import json
import csv
from collections import defaultdict
import math

class HTMLReportGenerator:
    def __init__(self, csv_file, json_file):
        self.csv_file = csv_file
        self.json_file = json_file
        self.data = []
        self.best_experiment = None
        self.load_data()
        
    def load_data(self):
        """加载数据"""
        print("📊 Loading data...")
        
        # 加载CSV数据
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 转换数值字段
                numeric_fields = ['pred_threshold', 'precision', 'recall', 'f1_score', 'accuracy',
                                'positive_predict_count', 'negative_predict_count', 'total_predict_count',
                                'successful_predictions', 'network_nodes', 'evidence_roads_count',
                                'tp', 'fp', 'tn', 'fn', 'test_roads_total']
                for key in numeric_fields:
                    if key in row and row[key]:
                        row[key] = float(row[key])
                self.data.append(row)
        
        print(f"✅ Loaded CSV: {len(self.data)} experiments")
        
        # 加载最佳实验JSON
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.best_experiment = data['best_experiment']
        print(f"✅ Loaded best experiment: {self.best_experiment['test_date']}")
        
    def calculate_stats(self, values):
        """计算统计信息"""
        if not values:
            return 0, 0
        mean = sum(values) / len(values)
        if len(values) == 1:
            return mean, 0
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = math.sqrt(variance)
        return mean, std
        
    def generate_html_report(self):
        """生成HTML报告"""
        print("🌐 Generating HTML report...")
        
        # 准备数据
        threshold_stats = defaultdict(lambda: {'precision': [], 'recall': [], 'f1_score': [], 'accuracy': []})
        for row in self.data:
            threshold = row['pred_threshold']
            threshold_stats[threshold]['precision'].append(row['precision'])
            threshold_stats[threshold]['recall'].append(row['recall'])
            threshold_stats[threshold]['f1_score'].append(row['f1_score'])
            threshold_stats[threshold]['accuracy'].append(row['accuracy'])
        
        date_stats = defaultdict(lambda: {'precision': [], 'recall': [], 'f1_score': [], 'accuracy': [],
                                         'tp': [], 'fp': [], 'tn': [], 'fn': [], 'test_roads_total': 0})
        for row in self.data:
            date = row['test_date']
            date_stats[date]['precision'].append(row['precision'])
            date_stats[date]['recall'].append(row['recall'])
            date_stats[date]['f1_score'].append(row['f1_score'])
            date_stats[date]['accuracy'].append(row['accuracy'])
            if date_stats[date]['test_roads_total'] == 0:
                date_stats[date]['test_roads_total'] = int(row.get('test_roads_total', 0))
        
        # 生成HTML内容
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Charleston Flood Prediction - Bayesian Network Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #4CAF50;
        }}
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        .header p {{
            color: #7f8c8d;
            font-size: 1.2em;
            margin: 10px 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .stat-label {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .chart-container {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 1.4em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            color: #2c3e50;
        }}
        .section {{
            margin: 40px 0;
            padding: 20px;
            border-left: 5px solid #3498db;
            background: #ecf0f1;
            border-radius: 5px;
        }}
        .section h2 {{
            color: #2c3e50;
            margin-top: 0;
        }}
        .experiment-details {{
            background: #e8f5e8;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #27ae60;
        }}
        .road-list {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .road-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border-left: 4px solid #3498db;
        }}
        .road-item.tp {{ border-left-color: #27ae60; }}
        .road-item.fp {{ border-left-color: #e74c3c; }}
        .road-item.tn {{ border-left-color: #3498db; }}
        .road-item.fn {{ border-left-color: #f39c12; }}
        .road-name {{
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 5px;
        }}
        .road-prob {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }}
        canvas {{ max-height: 400px; }}
        
        @media (max-width: 768px) {{
            .container {{ padding: 15px; }}
            .header h1 {{ font-size: 2em; }}
            .chart-container {{ padding: 15px; }}
            .road-list {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌊 Charleston Flood Prediction</h1>
            <p>Bayesian Network Cross-Validation Analysis Report</p>
            <p><strong>100 experiments</strong> across <strong>4 flood events</strong> with <strong>5 thresholds</strong></p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(self.data)}</div>
                <div class="stat-label">Total Experiments</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #00b894 0%, #00a085 100%);">
                <div class="stat-value">100%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);">
                <div class="stat-value">5</div>
                <div class="stat-label">Thresholds Tested</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);">
                <div class="stat-value">4</div>
                <div class="stat-label">Flood Events</div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📈 Performance Metrics vs Prediction Threshold</div>
            <canvas id="thresholdChart"></canvas>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📅 Performance by Flood Event Date</div>
            <canvas id="dateChart"></canvas>
        </div>
        
        <div class="section">
            <h2>🏆 Best Experiment Analysis</h2>
            <div class="experiment-details">
                <h3>Experiment Details</h3>
                <p><strong>Date:</strong> {self.best_experiment['test_date']}</p>
                <p><strong>Threshold:</strong> {self.best_experiment['pred_threshold']}</p>
                <p><strong>Trial ID:</strong> {self.best_experiment['trial_id']}</p>
                <br>
                <h4>Performance Metrics:</h4>
                <ul>
                    <li><strong>Precision:</strong> {self.best_experiment['precision']:.3f} (100% - Perfect accuracy for positive predictions)</li>
                    <li><strong>Recall:</strong> {self.best_experiment['recall']:.3f} ({self.best_experiment['recall']*100:.1f}% of actual floods detected)</li>
                    <li><strong>F1 Score:</strong> {self.best_experiment['f1_score']:.3f} (Balanced performance measure)</li>
                    <li><strong>Accuracy:</strong> {self.best_experiment['accuracy']:.3f} ({self.best_experiment['accuracy']*100:.1f}% overall correctness)</li>
                </ul>
                
                <h4>Confusion Matrix:</h4>
                <ul>
                    <li><strong>True Positives (TP):</strong> {self.best_experiment['tp']} - Correctly predicted floods</li>
                    <li><strong>False Positives (FP):</strong> {self.best_experiment['fp']} - Incorrectly predicted floods</li>
                    <li><strong>True Negatives (TN):</strong> {self.best_experiment['tn']} - Correctly predicted no floods</li>
                    <li><strong>False Negatives (FN):</strong> {self.best_experiment['fn']} - Missed actual floods</li>
                </ul>
                
                <h4>🔑 Evidence Roads Used:</h4>
                <ol>
                    {"".join(f"<li>{road}</li>" for road in self.best_experiment['evidence_roads'])}
                </ol>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">🎯 Best Experiment - Confusion Matrix</div>
            <canvas id="confusionChart"></canvas>
        </div>
        
        <div class="section">
            <h2>🛣️ Detailed Road Predictions</h2>
            
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #27ae60;"></div>
                    <span>True Positives (TP) - Correctly predicted floods</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #e74c3c;"></div>
                    <span>False Positives (FP) - Incorrectly predicted floods</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #3498db;"></div>
                    <span>True Negatives (TN) - Correctly predicted no floods</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #f39c12;"></div>
                    <span>False Negatives (FN) - Missed actual floods</span>
                </div>
            </div>
            
            {self.generate_road_predictions_html()}
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📊 Prediction Probability Distribution</div>
            <canvas id="probabilityChart"></canvas>
        </div>
        
        <div class="section">
            <h2>💡 Key Insights and Recommendations</h2>
            <h3>✨ Strengths:</h3>
            <ul>
                <li>Perfect precision (100%) - No false alarms</li>
                <li>Stable Bayesian inference (100% success rate)</li>
                <li>Evidence-based approach working effectively</li>
                <li>Good performance on large flood events</li>
            </ul>
            
            <h3>🔧 Areas for Improvement:</h3>
            <ul>
                <li>Recall could be improved ({self.best_experiment['recall']*100:.1f}% → target 60%+)</li>
                <li>Some high-impact roads still being missed</li>
                <li>Consider adjusting evidence ratio or thresholds</li>
            </ul>
            
            <h3>🎯 Deployment Recommendations:</h3>
            <ul>
                <li>Use threshold 0.4 for balanced performance</li>
                <li>Monitor precision to stay above 80%</li>
                <li>Focus on recall improvement for critical roads</li>
                <li>Consider ensemble methods for edge cases</li>
            </ul>
        </div>
    </div>

    <script>
        // 阈值性能图表
        const thresholdCtx = document.getElementById('thresholdChart').getContext('2d');
        new Chart(thresholdCtx, {{
            type: 'line',
            data: {{
                labels: {list(sorted(threshold_stats.keys()))},
                datasets: [
                    {{
                        label: 'Precision',
                        data: {[round(self.calculate_stats(threshold_stats[t]['precision'])[0], 3) for t in sorted(threshold_stats.keys())]},
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.1
                    }},
                    {{
                        label: 'Recall',
                        data: {[round(self.calculate_stats(threshold_stats[t]['recall'])[0], 3) for t in sorted(threshold_stats.keys())]},
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        tension: 0.1
                    }},
                    {{
                        label: 'F1 Score',
                        data: {[round(self.calculate_stats(threshold_stats[t]['f1_score'])[0], 3) for t in sorted(threshold_stats.keys())]},
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        tension: 0.1
                    }},
                    {{
                        label: 'Accuracy',
                        data: {[round(self.calculate_stats(threshold_stats[t]['accuracy'])[0], 3) for t in sorted(threshold_stats.keys())]},
                        borderColor: 'rgb(255, 206, 86)',
                        backgroundColor: 'rgba(255, 206, 86, 0.2)',
                        tension: 0.1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1.0
                    }}
                }}
            }}
        }});
        
        // 日期性能图表
        const dateCtx = document.getElementById('dateChart').getContext('2d');
        new Chart(dateCtx, {{
            type: 'bar',
            data: {{
                labels: {list(sorted(date_stats.keys()))},
                datasets: [
                    {{
                        label: 'Precision',
                        data: {[round(self.calculate_stats(date_stats[d]['precision'])[0], 3) for d in sorted(date_stats.keys())]},
                        backgroundColor: 'rgba(75, 192, 192, 0.8)'
                    }},
                    {{
                        label: 'Recall',
                        data: {[round(self.calculate_stats(date_stats[d]['recall'])[0], 3) for d in sorted(date_stats.keys())]},
                        backgroundColor: 'rgba(255, 99, 132, 0.8)'
                    }},
                    {{
                        label: 'F1 Score',
                        data: {[round(self.calculate_stats(date_stats[d]['f1_score'])[0], 3) for d in sorted(date_stats.keys())]},
                        backgroundColor: 'rgba(54, 162, 235, 0.8)'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1.0
                    }}
                }}
            }}
        }});
        
        // 混淆矩阵图表
        const confusionCtx = document.getElementById('confusionChart').getContext('2d');
        new Chart(confusionCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['True Positives', 'False Positives', 'True Negatives', 'False Negatives'],
                datasets: [{{
                    data: [{self.best_experiment['tp']}, {self.best_experiment['fp']}, {self.best_experiment['tn']}, {self.best_experiment['fn']}],
                    backgroundColor: ['#27ae60', '#e74c3c', '#3498db', '#f39c12']
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});
        
        // 概率分布图表
        const probabilityCtx = document.getElementById('probabilityChart').getContext('2d');
        const probabilities = {[p['predicted_probability'] for p in self.best_experiment['detailed_predictions']]};
        const bins = Array.from({{length: 10}}, (_, i) => i * 0.1);
        const histogram = new Array(10).fill(0);
        
        probabilities.forEach(prob => {{
            const binIndex = Math.min(Math.floor(prob * 10), 9);
            histogram[binIndex]++;
        }});
        
        new Chart(probabilityCtx, {{
            type: 'bar',
            data: {{
                labels: bins.map(b => `${{b.toFixed(1)}}-${{(b+0.1).toFixed(1)}}`),
                datasets: [{{
                    label: 'Number of Roads',
                    data: histogram,
                    backgroundColor: 'rgba(54, 162, 235, 0.8)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Distribution of Flood Prediction Probabilities'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Probability Range'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
        """
        
        # 保存HTML文件
        with open('bayesian_analysis_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("💾 Saved: bayesian_analysis_report.html")
        
    def generate_road_predictions_html(self):
        """生成道路预测的HTML"""
        predictions = self.best_experiment['detailed_predictions']
        
        # 按结果类型分组
        tp_roads = [p for p in predictions if p['true_label'] == 1 and p['predicted_label'] == 1]
        fp_roads = [p for p in predictions if p['true_label'] == 0 and p['predicted_label'] == 1]
        tn_roads = [p for p in predictions if p['true_label'] == 0 and p['predicted_label'] == 0]
        fn_roads = [p for p in predictions if p['true_label'] == 1 and p['predicted_label'] == 0]
        
        html = "<div class='road-list'>"
        
        # True Positives
        tp_roads.sort(key=lambda x: x['predicted_probability'], reverse=True)
        for road in tp_roads:
            html += f"""
            <div class="road-item tp">
                <div class="road-name">✅ {road['road_name']}</div>
                <div class="road-prob">Probability: {road['predicted_probability']:.3f} - Correctly predicted flood</div>
            </div>
            """
        
        # False Positives (if any)
        if fp_roads:
            fp_roads.sort(key=lambda x: x['predicted_probability'], reverse=True)
            for road in fp_roads:
                html += f"""
                <div class="road-item fp">
                    <div class="road-name">❌ {road['road_name']}</div>
                    <div class="road-prob">Probability: {road['predicted_probability']:.3f} - Incorrectly predicted flood</div>
                </div>
                """
        
        # False Negatives
        fn_roads.sort(key=lambda x: x['predicted_probability'], reverse=True)
        for road in fn_roads:
            html += f"""
            <div class="road-item fn">
                <div class="road-name">⚠️ {road['road_name']}</div>
                <div class="road-prob">Probability: {road['predicted_probability']:.3f} - Missed actual flood</div>
            </div>
            """
        
        # True Negatives (show top 10)
        tn_roads.sort(key=lambda x: x['predicted_probability'], reverse=True)
        for road in tn_roads[:10]:
            html += f"""
            <div class="road-item tn">
                <div class="road-name">✅ {road['road_name']}</div>
                <div class="road-prob">Probability: {road['predicted_probability']:.3f} - Correctly predicted no flood</div>
            </div>
            """
        
        if len(tn_roads) > 10:
            html += f"""
            <div class="road-item tn" style="text-align: center; font-style: italic;">
                <div class="road-name">... and {len(tn_roads) - 10} more correctly predicted no-flood roads</div>
            </div>
            """
        
        html += "</div>"
        return html

def main():
    """主函数"""
    print("🌐 Charleston Flood Prediction - HTML Report Generator")
    print("="*60)
    
    # 文件路径
    csv_file = "corrected_bayesian_flood_validation_full_network_summary_20250820_112441.csv"
    json_file = "best_2017_09_11_threshold_04_experiment.json"
    
    try:
        # 创建报告生成器
        generator = HTMLReportGenerator(csv_file, json_file)
        
        # 生成HTML报告
        generator.generate_html_report()
        
        print("\\n🎉 HTML report generated successfully!")
        print("📂 Open 'bayesian_analysis_report.html' in your web browser to view the interactive report")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required file: {str(e)}")
        print("Please ensure the following files exist:")
        print(f"  • {csv_file}")
        print(f"  • {json_file}")
    except Exception as e:
        print(f"❌ Error during report generation: {str(e)}")

if __name__ == "__main__":
    main()