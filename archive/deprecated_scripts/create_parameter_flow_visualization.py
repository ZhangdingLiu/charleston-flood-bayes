#!/usr/bin/env python3
"""
Parameter Flow Visualization

Creates a flow diagram showing how different parameter combinations connect to 
performance outcomes, with better configurations highlighted in darker colors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_top_configurations(csv_path):
    """Load the top configurations data"""
    df = pd.read_csv(csv_path)
    return df

def create_parameter_flow_diagram(df, output_dir):
    """Create a parameter flow diagram showing connections from parameters to performance"""
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Define parameter information from experiment config
    param_info = {
        'occ_thr': {'values': [2, 3, 4, 5], 'display_name': 'Occurrence\nThreshold'},
        'edge_thr': {'values': [1, 2, 3], 'display_name': 'Edge\nThreshold'},
        'weight_thr': {'values': [0.2, 0.3, 0.4, 0.5], 'display_name': 'Weight\nThreshold'},
        'evidence_count': {'values': [1, 2, 3, 4], 'display_name': 'Evidence\nCount'},
        'pred_threshold': {'values': [0.1, 0.2, 0.3, 0.4, 0.5], 'display_name': 'Prediction\nThreshold'},
        'neg_pos_ratio': {'values': [1.0, 1.5, 2.0], 'display_name': 'Neg/Pos\nRatio'},
        'marginal_prob_threshold': {'values': [0.03, 0.05, 0.08], 'display_name': 'Marginal Prob\nThreshold'}
    }
    
    # Set up column positions
    param_names = list(param_info.keys())
    n_params = len(param_names)
    col_width = 18 / (n_params + 2)  # +2 for performance columns
    
    # Colors for performance levels
    def get_performance_color(value, metric):
        if metric == 'f1_score':
            if value >= 0.8: return '#d73027'  # Dark red for excellent
            elif value >= 0.7: return '#fc8d59'  # Orange for good
            elif value >= 0.6: return '#fee08b'  # Light orange for fair
            else: return '#e0e0e0'  # Gray for poor
        elif metric == 'precision':
            if value >= 0.9: return '#1a9850'  # Dark green
            elif value >= 0.8: return '#66bd63'  # Green
            elif value >= 0.7: return '#a6d96a'  # Light green
            else: return '#e0e0e0'
        else:  # recall
            if value >= 0.9: return '#762a83'  # Dark purple
            elif value >= 0.8: return '#9970ab'  # Purple
            elif value >= 0.7: return '#c2a5cf'  # Light purple
            else: return '#e0e0e0'
    
    # Draw parameter columns
    y_positions = {}
    for col_idx, param in enumerate(param_names):
        x_pos = col_idx * col_width + 1
        values = param_info[param]['values']
        display_name = param_info[param]['display_name']
        
        # Column header
        ax.text(x_pos, 10.5, display_name, ha='center', va='center', 
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        # Parameter values
        y_spacing = 9 / len(values)
        param_y_positions = {}
        
        for val_idx, value in enumerate(values):
            y_pos = 9.5 - val_idx * y_spacing
            param_y_positions[value] = y_pos
            
            # Count how many top configs use this value
            value_count = len(df[df[param] == value])
            
            # Color intensity based on usage in top configs
            alpha = 0.3 + 0.7 * (value_count / len(df)) if value_count > 0 else 0.1
            
            # Draw parameter value box
            rect = patches.Rectangle((x_pos - 0.4, y_pos - 0.2), 0.8, 0.4,
                                   linewidth=1, edgecolor='black', 
                                   facecolor='lightblue', alpha=alpha)
            ax.add_patch(rect)
            
            # Add text
            ax.text(x_pos, y_pos, str(value), ha='center', va='center', fontsize=9)
            
            # Add count annotation
            if value_count > 0:
                ax.text(x_pos + 0.5, y_pos, f'({value_count})', ha='left', va='center', 
                       fontsize=7, color='darkblue')
        
        y_positions[param] = param_y_positions
    
    # Draw performance columns
    performance_metrics = ['precision', 'recall', 'f1_score']
    perf_col_start = n_params * col_width + 1.5
    
    for perf_idx, metric in enumerate(performance_metrics):
        x_pos = perf_col_start + perf_idx * col_width
        
        # Column header
        display_name = metric.replace('_', ' ').title()
        ax.text(x_pos, 10.5, display_name, ha='center', va='center', 
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        
        # Draw performance outcomes for top configurations
        for config_idx, (_, config) in enumerate(df.head(10).iterrows()):
            y_pos = 9.5 - config_idx * 0.8
            value = config[metric]
            color = get_performance_color(value, metric)
            
            # Draw performance box
            rect = patches.Rectangle((x_pos - 0.4, y_pos - 0.15), 0.8, 0.3,
                                   linewidth=1, edgecolor='black', 
                                   facecolor=color, alpha=0.8)
            ax.add_patch(rect)
            
            # Add performance value
            ax.text(x_pos, y_pos, f'{value:.3f}', ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='white' if value > 0.7 else 'black')
    
    # Draw connections for top 5 configurations
    top_configs = df.head(5)
    
    for config_idx, (_, config) in enumerate(top_configs.iterrows()):
        y_pos = 9.5 - config_idx * 0.8
        
        # Determine line color based on F1 score
        f1_score = config['f1_score']
        if f1_score >= 0.8:
            line_color = '#d73027'
            line_width = 3
            alpha = 0.8
        elif f1_score >= 0.7:
            line_color = '#fc8d59'
            line_width = 2
            alpha = 0.6
        else:
            line_color = '#999999'
            line_width = 1
            alpha = 0.4
        
        # Draw connections from each parameter to performance
        prev_x = None
        for col_idx, param in enumerate(param_names):
            x_pos = col_idx * col_width + 1
            param_value = config[param]
            param_y = y_positions[param][param_value]
            
            if prev_x is not None:
                # Draw connecting line between parameters
                ax.plot([prev_x + 0.4, x_pos - 0.4], [prev_y, param_y], 
                       color=line_color, linewidth=line_width, alpha=alpha)
            
            prev_x = x_pos
            prev_y = param_y
        
        # Connect to performance metrics
        for perf_idx, metric in enumerate(performance_metrics):
            perf_x = perf_col_start + perf_idx * col_width
            ax.plot([prev_x + 0.4, perf_x - 0.4], [prev_y, y_pos], 
                   color=line_color, linewidth=line_width, alpha=alpha)
    
    # Add title and labels
    ax.set_title('Parameter Flow Diagram: From Configuration to Performance\n' +
                'Top 5 Configurations Highlighted with Darker Lines', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], color='#d73027', lw=3, label='Excellent (F1 ≥ 0.8)'),
        Line2D([0], [0], color='#fc8d59', lw=2, label='Good (F1 ≥ 0.7)'),
        Line2D([0], [0], color='#999999', lw=1, label='Fair (F1 < 0.7)'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.15))
    
    # Add parameter range information
    info_text = "Parameter Ranges Tested:\n"
    for param, info in param_info.items():
        values_str = ", ".join(map(str, info['values']))
        info_text += f"• {param}: [{values_str}]\n"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
           facecolor='lightyellow', alpha=0.8))
    
    # Set axis limits and remove axes
    ax.set_xlim(0, perf_col_start + 3 * col_width)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Parameter flow diagram created successfully!")

def create_parameter_impact_heatmap(df, output_dir):
    """Create a heatmap showing parameter impact on performance"""
    
    # Read the original filtered data to get more samples
    results_dir = Path("results/parameter_optimization_20250721_140722")
    csv_path = results_dir / "complete_results.csv"
    full_df = pd.read_csv(csv_path)
    
    # Apply filtering
    filtered_df = full_df[
        (full_df['negative_candidates_count'] >= 9) &
        (full_df['precision'] < 1.0) &
        (full_df['recall'] < 1.0)
    ].copy()
    
    # Create parameter impact analysis
    params = ['occ_thr', 'edge_thr', 'weight_thr', 'evidence_count', 
              'pred_threshold', 'neg_pos_ratio', 'marginal_prob_threshold']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Parameter Impact on Performance Metrics', fontsize=16, fontweight='bold')
    
    metrics = ['precision', 'recall', 'f1_score']
    metric_names = ['Precision', 'Recall', 'F1 Score']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        # Create impact matrix
        impact_data = []
        param_labels = []
        
        for param in params:
            param_impacts = []
            unique_values = sorted(filtered_df[param].unique())
            
            for value in unique_values:
                avg_performance = filtered_df[filtered_df[param] == value][metric].mean()
                param_impacts.append(avg_performance)
            
            # Pad with NaN if needed to make all rows same length
            max_len = max(len(filtered_df[p].unique()) for p in params)
            while len(param_impacts) < max_len:
                param_impacts.append(np.nan)
                
            impact_data.append(param_impacts)
            param_labels.append(param.replace('_', '\n'))
        
        # Create heatmap
        impact_matrix = np.array(impact_data)
        
        im = axes[i].imshow(impact_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        # Add value annotations
        for row in range(len(param_labels)):
            for col in range(impact_matrix.shape[1]):
                if not np.isnan(impact_matrix[row, col]):
                    text = axes[i].text(col, row, f'{impact_matrix[row, col]:.3f}',
                                      ha="center", va="center", color="black", fontsize=8)
        
        axes[i].set_title(f'{metric_name} by Parameter Values')
        axes[i].set_yticks(range(len(param_labels)))
        axes[i].set_yticklabels(param_labels, fontsize=9)
        axes[i].set_xlabel('Parameter Value Index')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_impact_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Parameter impact heatmap created successfully!")

def main():
    """Main function to create parameter flow visualizations"""
    
    # Paths
    results_dir = Path("results/parameter_optimization_20250721_140722")
    analysis_dir = results_dir / "analysis"
    top_configs_path = analysis_dir / "top_configurations_summary.csv"
    
    print("Creating parameter flow visualizations...")
    
    # Load top configurations
    df = load_top_configurations(top_configs_path)
    
    print(f"Loaded {len(df)} top configurations")
    
    # Create visualizations
    create_parameter_flow_diagram(df, analysis_dir)
    create_parameter_impact_heatmap(df, analysis_dir)
    
    print(f"\n✅ Parameter flow visualizations completed!")
    print(f"📁 Saved to: {analysis_dir}")
    print("📊 Generated files:")
    print("   - parameter_flow_diagram.png")
    print("   - parameter_impact_heatmap.png")

if __name__ == "__main__":
    main()