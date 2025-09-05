#!/usr/bin/env python3
"""
Enhanced Parameter Flow Visualization

Creates a comprehensive parameter flow diagram showing more configurations
including top performers, medium performers, and poor performers to give
a complete view of the parameter space and performance distribution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_curved_connection(ax, start_pos, end_pos, color='blue', linewidth=2, alpha=0.7):
    """Create a curved connection between two points"""
    x1, y1 = start_pos
    x2, y2 = end_pos
    
    # Calculate control points for smooth curve
    mid_x = (x1 + x2) / 2
    ctrl1_x = x1 + (mid_x - x1) * 0.3
    ctrl2_x = x2 - (x2 - mid_x) * 0.3
    
    # Create path
    verts = [
        (x1, y1),           # Start point
        (ctrl1_x, y1),      # Control point 1
        (ctrl2_x, y2),      # Control point 2
        (x2, y2),           # End point
    ]
    
    codes = [mpath.Path.MOVETO,
             mpath.Path.CURVE4,
             mpath.Path.CURVE4,
             mpath.Path.CURVE4]
    
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor='none', edgecolor=color, 
                              linewidth=linewidth, alpha=alpha)
    ax.add_patch(patch)

def select_diverse_configurations(df):
    """Select a diverse set of configurations including excellent, good, fair, and poor performers"""
    
    # Sort by F1 score
    df_sorted = df.sort_values('f1_score', ascending=False)
    
    selected_configs = []
    
    # 1. Top 3 performers (Excellent: F1 >= 0.8)
    excellent = df_sorted[df_sorted['f1_score'] >= 0.8].head(3)
    for i, (_, config) in enumerate(excellent.iterrows()):
        config_info = config.copy()
        config_info['performance_tier'] = 'Excellent'
        config_info['display_label'] = f'Top {i+1}'
        selected_configs.append(config_info)
    
    # 2. Good performers (0.7 <= F1 < 0.8)
    good = df_sorted[(df_sorted['f1_score'] >= 0.7) & (df_sorted['f1_score'] < 0.8)]
    if len(good) > 0:
        # Select 2-3 good performers with different parameter patterns
        good_selected = good.head(3)
        for i, (_, config) in enumerate(good_selected.iterrows()):
            config_info = config.copy()
            config_info['performance_tier'] = 'Good'
            config_info['display_label'] = f'Good {i+1}'
            selected_configs.append(config_info)
    
    # 3. Fair performers (0.5 <= F1 < 0.7)
    fair = df_sorted[(df_sorted['f1_score'] >= 0.5) & (df_sorted['f1_score'] < 0.7)]
    if len(fair) > 0:
        # Select 2 fair performers
        fair_selected = fair.head(2)
        for i, (_, config) in enumerate(fair_selected.iterrows()):
            config_info = config.copy()
            config_info['performance_tier'] = 'Fair'
            config_info['display_label'] = f'Fair {i+1}'
            selected_configs.append(config_info)
    
    # 4. Poor performers (F1 < 0.5)
    poor = df_sorted[df_sorted['f1_score'] < 0.5]
    if len(poor) > 0:
        # Select 2 poor performers to show what doesn't work
        poor_selected = poor.head(2)
        for i, (_, config) in enumerate(poor_selected.iterrows()):
            config_info = config.copy()
            config_info['performance_tier'] = 'Poor'
            config_info['display_label'] = f'Poor {i+1}'
            selected_configs.append(config_info)
    
    return pd.DataFrame(selected_configs)

def create_enhanced_parameter_flow_diagram(df, output_dir):
    """Create an enhanced parameter flow diagram with diverse performance examples"""
    
    # Load complete filtered data for better selection
    results_dir = Path("results/parameter_optimization_20250721_140722")
    csv_path = results_dir / "complete_results.csv"
    full_df = pd.read_csv(csv_path)
    
    # Apply filtering
    filtered_df = full_df[
        (full_df['negative_candidates_count'] >= 9) &
        (full_df['precision'] < 1.0) &
        (full_df['recall'] < 1.0)
    ].copy()
    
    # Select diverse configurations
    selected_configs = select_diverse_configurations(filtered_df)
    
    print(f"Selected {len(selected_configs)} diverse configurations:")
    for tier in ['Excellent', 'Good', 'Fair', 'Poor']:
        tier_configs = selected_configs[selected_configs['performance_tier'] == tier]
        if len(tier_configs) > 0:
            f1_range = f"{tier_configs['f1_score'].min():.3f}-{tier_configs['f1_score'].max():.3f}"
            print(f"  - {tier}: {len(tier_configs)} configs (F1: {f1_range})")
    
    # Set up larger figure for more configurations
    fig, ax = plt.subplots(figsize=(26, 14))
    
    # Define parameter information
    param_info = {
        'occ_thr': {'values': [2, 3, 4, 5], 'display_name': 'Occurrence\nThreshold'},
        'edge_thr': {'values': [1, 2, 3], 'display_name': 'Edge\nThreshold'},
        'weight_thr': {'values': [0.2, 0.3, 0.4, 0.5], 'display_name': 'Weight\nThreshold'},
        'evidence_count': {'values': [1, 2, 3, 4], 'display_name': 'Evidence\nCount'},
        'pred_threshold': {'values': [0.1, 0.2, 0.3, 0.4, 0.5], 'display_name': 'Prediction\nThreshold'},
        'neg_pos_ratio': {'values': [1.0, 1.5, 2.0], 'display_name': 'Neg/Pos\nRatio'},
        'marginal_prob_threshold': {'values': [0.03, 0.05, 0.08], 'display_name': 'Marginal Prob\nThreshold'}
    }
    
    # Better column positioning
    param_names = list(param_info.keys())
    n_params = len(param_names)
    total_width = 22
    param_width = 14
    perf_width = 6
    col_spacing = param_width / n_params
    
    # Performance tier colors and line styles
    tier_styles = {
        'Excellent': {'color': '#d73027', 'linewidth': 4, 'alpha': 0.9, 'linestyle': '-'},
        'Good': {'color': '#fc8d59', 'linewidth': 3, 'alpha': 0.8, 'linestyle': '-'},
        'Fair': {'color': '#fee08b', 'linewidth': 2, 'alpha': 0.7, 'linestyle': '--'},
        'Poor': {'color': '#999999', 'linewidth': 1.5, 'alpha': 0.6, 'linestyle': ':'}
    }
    
    # Performance metrics with improved colors
    performance_colors = {
        'precision': {'excellent': '#1b7837', 'good': '#5aae61', 'fair': '#a6dba0', 'poor': '#f7f7f7'},
        'recall': {'excellent': '#762a83', 'good': '#9970ab', 'fair': '#c2a5cf', 'poor': '#f7f7f7'},
        'f1_score': {'excellent': '#b2182b', 'good': '#d6604d', 'fair': '#f4a582', 'poor': '#f7f7f7'}
    }
    
    def get_performance_color_enhanced(value, metric):
        colors = performance_colors[metric]
        if value >= 0.8: return colors['excellent']
        elif value >= 0.7: return colors['good'] 
        elif value >= 0.5: return colors['fair']
        else: return colors['poor']
    
    # Draw parameter columns
    y_positions = {}
    for col_idx, param in enumerate(param_names):
        x_pos = 2 + col_idx * col_spacing
        values = param_info[param]['values']
        display_name = param_info[param]['display_name']
        
        # Column header
        header_box = FancyBboxPatch((x_pos - 0.6, 11.8), 1.2, 0.8,
                                   boxstyle="round,pad=0.1", 
                                   facecolor='#4472C4', edgecolor='black',
                                   alpha=0.8)
        ax.add_patch(header_box)
        ax.text(x_pos, 12.2, display_name, ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
        
        # Parameter values
        param_y_positions = {}
        y_start = 11.0
        y_spacing = 10.0 / len(values)
        
        for val_idx, value in enumerate(values):
            y_pos = y_start - val_idx * y_spacing
            param_y_positions[value] = y_pos
            
            # Count usage in selected configs
            value_count = len(selected_configs[selected_configs[param] == value])
            alpha = 0.4 + 0.6 * (value_count / max(1, len(selected_configs)))
            
            # Parameter value box
            param_box = FancyBboxPatch((x_pos - 0.4, y_pos - 0.25), 0.8, 0.5,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#B4C7E7', edgecolor='#2F5597',
                                      alpha=alpha, linewidth=1.5)
            ax.add_patch(param_box)
            
            # Parameter value text
            ax.text(x_pos, y_pos, str(value), ha='center', va='center', 
                   fontsize=9, fontweight='bold')
        
        y_positions[param] = param_y_positions
    
    # Draw performance columns
    performance_metrics = ['precision', 'recall', 'f1_score']
    metric_display_names = ['Precision', 'Recall', 'F1 Score']
    perf_col_start = 2 + n_params * col_spacing + 1
    perf_col_spacing = perf_width / len(performance_metrics)
    
    for perf_idx, (metric, display_name) in enumerate(zip(performance_metrics, metric_display_names)):
        x_pos = perf_col_start + perf_idx * perf_col_spacing
        
        # Performance column header
        header_box = FancyBboxPatch((x_pos - 0.6, 11.8), 1.2, 0.8,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#70AD47', edgecolor='black',
                                   alpha=0.8)
        ax.add_patch(header_box)
        ax.text(x_pos, 12.2, display_name, ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
        
        # Performance outcomes for all selected configurations
        for config_idx, (_, config) in enumerate(selected_configs.iterrows()):
            y_pos = 10.7 - config_idx * 1.0
            value = config[metric]
            color = get_performance_color_enhanced(value, metric)
            
            # Performance box
            perf_box = FancyBboxPatch((x_pos - 0.45, y_pos - 0.2), 0.9, 0.4,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black',
                                     alpha=0.9, linewidth=1)
            ax.add_patch(perf_box)
            
            # Performance value
            text_color = 'white' if value > 0.5 else 'black'
            ax.text(x_pos, y_pos, f'{value:.3f}', ha='center', va='center', 
                   fontsize=8, fontweight='bold', color=text_color)
    
    # Draw connections for all selected configurations
    for config_idx, (_, config) in enumerate(selected_configs.iterrows()):
        y_pos = 10.7 - config_idx * 1.0
        tier = config['performance_tier']
        label = config['display_label']
        style = tier_styles[tier]
        
        # Configuration label
        label_color = style['color']
        ax.text(0.5, y_pos, label, ha='center', va='center',
               fontsize=9, fontweight='bold', color=label_color,
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                        edgecolor=label_color, alpha=0.8))
        
        # Draw connections through parameters
        connection_points = []
        
        for col_idx, param in enumerate(param_names):
            x_pos = 2 + col_idx * col_spacing
            param_value = config[param]
            param_y = y_positions[param][param_value]
            connection_points.append((x_pos, param_y))
        
        # Connect parameter points
        for i in range(len(connection_points) - 1):
            start_pos = (connection_points[i][0] + 0.4, connection_points[i][1])
            end_pos = (connection_points[i + 1][0] - 0.4, connection_points[i + 1][1])
            
            # Create line with appropriate style
            if style['linestyle'] == '--':
                line = ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                              color=style['color'], linewidth=style['linewidth'], 
                              alpha=style['alpha'], linestyle='--')[0]
            elif style['linestyle'] == ':':
                line = ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                              color=style['color'], linewidth=style['linewidth'], 
                              alpha=style['alpha'], linestyle=':')[0]
            else:
                create_curved_connection(ax, start_pos, end_pos, color=style['color'],
                                      linewidth=style['linewidth'], alpha=style['alpha'])
        
        # Connect to performance columns
        last_param_pos = (connection_points[-1][0] + 0.4, connection_points[-1][1])
        for perf_idx in range(len(performance_metrics)):
            perf_x = perf_col_start + perf_idx * perf_col_spacing
            perf_pos = (perf_x - 0.45, y_pos)
            
            if style['linestyle'] == '--':
                ax.plot([last_param_pos[0], perf_pos[0]], [last_param_pos[1], perf_pos[1]], 
                       color=style['color'], linewidth=style['linewidth'], 
                       alpha=style['alpha'], linestyle='--')
            elif style['linestyle'] == ':':
                ax.plot([last_param_pos[0], perf_pos[0]], [last_param_pos[1], perf_pos[1]], 
                       color=style['color'], linewidth=style['linewidth'], 
                       alpha=style['alpha'], linestyle=':')
            else:
                create_curved_connection(ax, last_param_pos, perf_pos, color=style['color'],
                                      linewidth=style['linewidth'], alpha=style['alpha'])
    
    # Add title
    ax.text(13, 13.0, 'Enhanced Parameter Flow Diagram: Diverse Configuration Examples', 
           ha='center', va='center', fontsize=18, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.3))
    
    # Add subtitle
    ax.text(13, 12.5, f'Showing {len(selected_configs)} Configurations Across Performance Spectrum', 
           ha='center', va='center', fontsize=14, style='italic')
    
    # Create comprehensive legend
    legend_elements = []
    for tier, style in tier_styles.items():
        tier_configs = selected_configs[selected_configs['performance_tier'] == tier]
        if len(tier_configs) > 0:
            f1_range = f"(F1: {tier_configs['f1_score'].min():.3f}-{tier_configs['f1_score'].max():.3f})"
            label = f"{tier} {f1_range}"
            
            if style['linestyle'] == '--':
                legend_elements.append(plt.Line2D([0], [0], color=style['color'], 
                                                 lw=style['linewidth'], linestyle='--', label=label))
            elif style['linestyle'] == ':':
                legend_elements.append(plt.Line2D([0], [0], color=style['color'], 
                                                 lw=style['linewidth'], linestyle=':', label=label))
            else:
                legend_elements.append(plt.Line2D([0], [0], color=style['color'], 
                                                 lw=style['linewidth'], label=label))
    
    legend = ax.legend(handles=legend_elements, loc='upper right', 
                      bbox_to_anchor=(0.98, 0.95), frameon=True,
                      fancybox=True, shadow=True, fontsize=11)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Add parameter ranges information (bottom-left)
    info_text = "Parameter Ranges:\n"
    for param, info in param_info.items():
        values_str = ", ".join(map(str, info['values']))
        display_name = info['display_name'].replace('\n', ' ')
        info_text += f"• {display_name}: {values_str}\n"
    
    ax.text(0.02, 0.35, info_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', 
           bbox=dict(boxstyle="round,pad=0.5", facecolor='#F2F2F2', 
                    edgecolor='gray', alpha=0.9))
    
    # Add performance interpretation guide
    perf_guide_text = "Performance Interpretation:\n"
    perf_guide_text += "• Excellent (F1≥0.8): Production-ready configurations\n"
    perf_guide_text += "• Good (0.7≤F1<0.8): Viable with tuning\n"
    perf_guide_text += "• Fair (0.5≤F1<0.7): Needs significant improvement\n"
    perf_guide_text += "• Poor (F1<0.5): Not recommended for deployment\n"
    
    ax.text(0.02, 0.02, perf_guide_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='bottom',
           bbox=dict(boxstyle="round,pad=0.4", facecolor='#FFF8DC', 
                    edgecolor='orange', alpha=0.8))
    
    # Add configuration count summary
    count_text = f"Configuration Summary:\n"
    for tier in ['Excellent', 'Good', 'Fair', 'Poor']:
        count = len(selected_configs[selected_configs['performance_tier'] == tier])
        if count > 0:
            count_text += f"• {tier}: {count} configurations\n"
    count_text += f"• Total shown: {len(selected_configs)} / {len(filtered_df)} filtered"
    
    ax.text(0.75, 0.02, count_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='bottom',
           bbox=dict(boxstyle="round,pad=0.4", facecolor='#E8F4FD', 
                    edgecolor='blue', alpha=0.8))
    
    # Set axis limits and remove axes
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 13.5)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'enhanced_parameter_flow_diagram.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Enhanced parameter flow diagram created successfully!")
    return selected_configs

def main():
    """Main function to create enhanced parameter flow visualization"""
    
    # Paths
    results_dir = Path("results/parameter_optimization_20250721_140722")
    analysis_dir = results_dir / "analysis"
    top_configs_path = analysis_dir / "top_configurations_summary.csv"
    
    print("Creating enhanced parameter flow visualization with diverse examples...")
    
    # Load top configurations (just for reference)
    df = pd.read_csv(top_configs_path)
    
    # Create enhanced visualization
    selected_configs = create_enhanced_parameter_flow_diagram(df, analysis_dir)
    
    print(f"\n✅ Enhanced visualization completed!")
    print(f"📁 Saved to: {analysis_dir}")
    print("📊 Generated: enhanced_parameter_flow_diagram.png")
    print(f"\n📈 Configuration diversity:")
    
    # Show summary of selected configurations
    for tier in ['Excellent', 'Good', 'Fair', 'Poor']:
        tier_configs = selected_configs[selected_configs['performance_tier'] == tier]
        if len(tier_configs) > 0:
            f1_scores = tier_configs['f1_score'].tolist()
            f1_str = ", ".join([f"{f1:.3f}" for f1 in f1_scores])
            print(f"   - {tier}: {len(tier_configs)} configs (F1: {f1_str})")
    
    print(f"\n🎨 Visual enhancements:")
    print("   - Shows performance spectrum from excellent to poor")
    print("   - Different line styles for different performance tiers")
    print("   - Comprehensive legend with F1 score ranges")
    print("   - Performance interpretation guide")
    print("   - Configuration count summaries")

if __name__ == "__main__":
    main()