#!/usr/bin/env python3
"""
Improved Parameter Flow Visualization

Creates a cleaner, more readable parameter flow diagram with better layout,
improved spacing, and clearer visual connections between parameters and performance.
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

def create_improved_parameter_flow_diagram(df, output_dir):
    """Create an improved parameter flow diagram with better layout and visual design"""
    
    # Set up larger figure with better proportions
    fig, ax = plt.subplots(figsize=(24, 10))
    
    # Define parameter information with better display names
    param_info = {
        'occ_thr': {'values': [2, 3, 4, 5], 'display_name': 'Occurrence\nThreshold'},
        'edge_thr': {'values': [1, 2, 3], 'display_name': 'Edge\nThreshold'},
        'weight_thr': {'values': [0.2, 0.3, 0.4, 0.5], 'display_name': 'Weight\nThreshold'},
        'evidence_count': {'values': [1, 2, 3, 4], 'display_name': 'Evidence\nCount'},
        'pred_threshold': {'values': [0.1, 0.2, 0.3, 0.4, 0.5], 'display_name': 'Prediction\nThreshold'},
        'neg_pos_ratio': {'values': [1.0, 1.5, 2.0], 'display_name': 'Neg/Pos\nRatio'},
        'marginal_prob_threshold': {'values': [0.03, 0.05, 0.08], 'display_name': 'Marginal Prob\nThreshold'}
    }
    
    # Better column positioning with more space
    param_names = list(param_info.keys())
    n_params = len(param_names)
    total_width = 20
    param_width = 12
    perf_width = 6
    col_spacing = param_width / n_params
    
    # Performance metrics with better colors
    performance_colors = {
        'precision': {'excellent': '#1b7837', 'good': '#5aae61', 'fair': '#a6dba0', 'poor': '#e0e0e0'},
        'recall': {'excellent': '#762a83', 'good': '#9970ab', 'fair': '#c2a5cf', 'poor': '#e0e0e0'},
        'f1_score': {'excellent': '#b2182b', 'good': '#d6604d', 'fair': '#f4a582', 'poor': '#e0e0e0'}
    }
    
    def get_performance_color_improved(value, metric):
        colors = performance_colors[metric]
        if value >= 0.85: return colors['excellent']
        elif value >= 0.75: return colors['good'] 
        elif value >= 0.65: return colors['fair']
        else: return colors['poor']
    
    # Draw parameter columns with improved styling
    y_positions = {}
    for col_idx, param in enumerate(param_names):
        x_pos = 2 + col_idx * col_spacing
        values = param_info[param]['values']
        display_name = param_info[param]['display_name']
        
        # Column header with better styling
        header_box = FancyBboxPatch((x_pos - 0.6, 8.3), 1.2, 0.8,
                                   boxstyle="round,pad=0.1", 
                                   facecolor='#4472C4', edgecolor='black',
                                   alpha=0.8)
        ax.add_patch(header_box)
        ax.text(x_pos, 8.7, display_name, ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
        
        # Parameter values with better spacing
        param_y_positions = {}
        y_start = 7.5
        y_spacing = 6.5 / len(values)
        
        for val_idx, value in enumerate(values):
            y_pos = y_start - val_idx * y_spacing
            param_y_positions[value] = y_pos
            
            # Count usage in top configs (for transparency effect)
            value_count = len(df[df[param] == value])
            alpha = 0.4 + 0.6 * (value_count / max(1, len(df)))
            
            # Parameter value box with rounded corners
            param_box = FancyBboxPatch((x_pos - 0.4, y_pos - 0.25), 0.8, 0.5,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#B4C7E7', edgecolor='#2F5597',
                                      alpha=alpha, linewidth=1.5)
            ax.add_patch(param_box)
            
            # Parameter value text
            ax.text(x_pos, y_pos, str(value), ha='center', va='center', 
                   fontsize=9, fontweight='bold')
        
        y_positions[param] = param_y_positions
    
    # Draw performance columns with improved design
    performance_metrics = ['precision', 'recall', 'f1_score']
    metric_display_names = ['Precision', 'Recall', 'F1 Score']
    perf_col_start = 2 + n_params * col_spacing + 1
    perf_col_spacing = perf_width / len(performance_metrics)
    
    # Select top 3 configurations for cleaner visualization
    top_configs = df.head(3)
    
    for perf_idx, (metric, display_name) in enumerate(zip(performance_metrics, metric_display_names)):
        x_pos = perf_col_start + perf_idx * perf_col_spacing
        
        # Performance column header
        header_box = FancyBboxPatch((x_pos - 0.6, 8.3), 1.2, 0.8,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#70AD47', edgecolor='black',
                                   alpha=0.8)
        ax.add_patch(header_box)
        ax.text(x_pos, 8.7, display_name, ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
        
        # Performance outcomes for top configurations
        for config_idx, (_, config) in enumerate(top_configs.iterrows()):
            y_pos = 7.2 - config_idx * 1.2
            value = config[metric]
            color = get_performance_color_improved(value, metric)
            
            # Performance box with better styling
            perf_box = FancyBboxPatch((x_pos - 0.45, y_pos - 0.3), 0.9, 0.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor='black',
                                     alpha=0.9, linewidth=1.5)
            ax.add_patch(perf_box)
            
            # Performance value with better text styling
            text_color = 'white' if value > 0.7 else 'black'
            ax.text(x_pos, y_pos, f'{value:.3f}', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color=text_color)
    
    # Draw curved connections for top 3 configurations
    config_colors = ['#d73027', '#fc8d59', '#1a9850']  # Red, orange, green
    config_names = ['Best F1', '2nd Best', '3rd Best']
    
    for config_idx, (_, config) in enumerate(top_configs.iterrows()):
        y_pos = 7.2 - config_idx * 1.2
        color = config_colors[config_idx]
        
        # Configuration label
        ax.text(0.5, y_pos, config_names[config_idx], ha='center', va='center',
               fontsize=10, fontweight='bold', color=color,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                        edgecolor=color, alpha=0.8))
        
        # Draw connections through parameters
        connection_points = []
        
        for col_idx, param in enumerate(param_names):
            x_pos = 2 + col_idx * col_spacing
            param_value = config[param]
            param_y = y_positions[param][param_value]
            connection_points.append((x_pos, param_y))
        
        # Connect parameter points with curves
        for i in range(len(connection_points) - 1):
            start_pos = (connection_points[i][0] + 0.4, connection_points[i][1])
            end_pos = (connection_points[i + 1][0] - 0.4, connection_points[i + 1][1])
            create_curved_connection(ax, start_pos, end_pos, color=color, 
                                  linewidth=3, alpha=0.8)
        
        # Connect to performance columns
        last_param_pos = (connection_points[-1][0] + 0.4, connection_points[-1][1])
        for perf_idx in range(len(performance_metrics)):
            perf_x = perf_col_start + perf_idx * perf_col_spacing
            perf_pos = (perf_x - 0.45, y_pos)
            create_curved_connection(ax, last_param_pos, perf_pos, color=color,
                                  linewidth=3, alpha=0.8)
    
    # Add title with better styling
    ax.text(12, 9.5, 'Parameter Flow Diagram: Configuration → Performance', 
           ha='center', va='center', fontsize=18, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.3))
    
    # Add subtitle
    ax.text(12, 9.0, 'Top 3 Configurations with Optimized Layout', 
           ha='center', va='center', fontsize=14, style='italic')
    
    # Create improved legend
    legend_elements = [
        plt.Line2D([0], [0], color=config_colors[0], lw=4, label='Best F1 Configuration'),
        plt.Line2D([0], [0], color=config_colors[1], lw=4, label='2nd Best Configuration'), 
        plt.Line2D([0], [0], color=config_colors[2], lw=4, label='3rd Best Configuration')
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper right', 
                      bbox_to_anchor=(0.98, 0.95), frameon=True,
                      fancybox=True, shadow=True, fontsize=11)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Add parameter ranges information in bottom-left (non-overlapping)
    info_text = "Parameter Ranges:\n"
    for param, info in param_info.items():
        values_str = ", ".join(map(str, info['values']))
        display_name = info['display_name'].replace('\n', ' ')
        info_text += f"• {display_name}: {values_str}\n"
    
    # Position at bottom-left, safely away from content
    ax.text(0.02, 0.25, info_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', 
           bbox=dict(boxstyle="round,pad=0.5", facecolor='#F2F2F2', 
                    edgecolor='gray', alpha=0.9))
    
    # Add performance color guide
    color_guide_text = "Performance Levels:\n"
    for metric, colors in performance_colors.items():
        metric_name = metric.replace('_', ' ').title()
        color_guide_text += f"{metric_name}: "
        color_guide_text += "●Excellent (≥0.85) ●Good (≥0.75) ●Fair (≥0.65) ●Poor (<0.65)\n"
    
    ax.text(0.02, 0.02, color_guide_text, transform=ax.transAxes, fontsize=8,
           verticalalignment='bottom',
           bbox=dict(boxstyle="round,pad=0.4", facecolor='#E7F3FF', 
                    edgecolor='blue', alpha=0.8))
    
    # Set axis limits and remove axes
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improved_parameter_flow_diagram.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Improved parameter flow diagram created successfully!")

def main():
    """Main function to create improved parameter flow visualization"""
    
    # Paths
    results_dir = Path("results/parameter_optimization_20250721_140722")
    analysis_dir = results_dir / "analysis"
    top_configs_path = analysis_dir / "top_configurations_summary.csv"
    
    print("Creating improved parameter flow visualization...")
    
    # Load top configurations
    df = pd.read_csv(top_configs_path)
    
    print(f"Loaded {len(df)} top configurations")
    
    # Create improved visualization
    create_improved_parameter_flow_diagram(df, analysis_dir)
    
    print(f"\n✅ Improved visualization completed!")
    print(f"📁 Saved to: {analysis_dir}")
    print("📊 Generated: improved_parameter_flow_diagram.png")
    print("\n🎨 Improvements made:")
    print("   - Moved parameter info to bottom-left (no overlap)")
    print("   - Removed count annotations from parameter values")
    print("   - Better spacing and cleaner layout")
    print("   - Curved connections for better flow visualization")
    print("   - Reduced to top 3 configs for clarity")
    print("   - Enhanced colors and styling")

if __name__ == "__main__":
    main()