import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import networkx as nx

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, '0_DataPreparation'))
from config import (TRAIN_PATH, VIZ_DIR, WARENGRUPPEN, get_warengruppe_name)

def save_correlation_results(correlations_dict, output_dir):
    """Save correlation analysis results to files"""
    # Create timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save correlations
    corr_file = os.path.join(output_dir, f'product_correlations_{timestamp}.txt')
    with open(corr_file, 'w') as f:
        f.write("Product Group Correlation Analysis\n")
        f.write("================================\n\n")
        for title, content in correlations_dict.items():
            f.write(f"{title}\n")
            f.write("-" * len(title) + "\n")
            f.write(f"{content}\n\n")
    
    return corr_file

def create_correlation_network(correlations_df, viz_output_dir):
    """Create and save network visualization of correlations"""
    # Create a new graph
    G = nx.Graph()
    
    # Add nodes (products)
    products = set(correlations_df['Product1'].unique()) | set(correlations_df['Product2'].unique())
    for product in products:
        G.add_node(product)
    
    # Add edges (correlations)
    for _, row in correlations_df.iterrows():
        G.add_edge(row['Product1'], 
                  row['Product2'], 
                  weight=abs(row['Correlation']),
                  correlation=row['Correlation'])
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Set up the layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=2000, alpha=0.7)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # Draw edges with different colors for positive/negative correlations
    edges_pos = [(u, v) for (u, v, d) in G.edges(data=True) if d['correlation'] > 0]
    edges_neg = [(u, v) for (u, v, d) in G.edges(data=True) if d['correlation'] < 0]
    
    # Get correlation values for edge widths
    edge_weights = [abs(d['correlation']) * 3 for (_, _, d) in G.edges(data=True)]
    
    # Draw positive correlations in red
    nx.draw_networkx_edges(G, pos, edgelist=edges_pos, 
                          edge_color='red', width=edge_weights,
                          alpha=0.6)
    
    # Draw negative correlations in blue
    nx.draw_networkx_edges(G, pos, edgelist=edges_neg,
                          edge_color='blue', width=edge_weights,
                          alpha=0.6)
    
    # Add correlation values as edge labels
    edge_labels = nx.get_edge_attributes(G, 'correlation')
    edge_labels = {k: f'{v:.2f}' for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title('Product Correlation Network\nRed: Positive Correlations, Blue: Negative Correlations\nEdge width represents correlation strength')
    plt.axis('off')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(viz_output_dir, 'correlation_network.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def analyze_correlations():
    """Analyze correlations between product groups"""
    print("Analyzing correlations between product groups...")
    
    # Create output directories
    output_dir = os.path.join(SCRIPT_DIR, 'output', 'correlation_analysis')
    viz_output_dir = os.path.join(VIZ_DIR, 'correlations')
    for directory in [output_dir, viz_output_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df['Datum'] = pd.to_datetime(train_df['Datum'])
    
    # Create pivot table for product group correlations
    print("Calculating correlations between product groups...")
    pivot_table = train_df.pivot_table(
        index='Datum',
        columns='Warengruppe',
        values='Umsatz',
        aggfunc='sum'
    )
    
    # Rename columns from numbers to names
    pivot_table.columns = [get_warengruppe_name(col) for col in pivot_table.columns]
    product_corr = pivot_table.corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(product_corr, 
                annot=True, 
                cmap='coolwarm', 
                center=0, 
                fmt='.2f',
                square=True)
    plt.title('Product Group Correlations')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_output_dir, 'product_correlations.png'), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    # Find strongest correlations
    correlations = []
    for i in range(len(product_corr.columns)):
        for j in range(i+1, len(product_corr.columns)):
            correlations.append({
                'Product1': product_corr.columns[i],
                'Product2': product_corr.columns[j],
                'Correlation': product_corr.iloc[i, j]
            })
    
    correlations_df = pd.DataFrame(correlations)
    
    # Create network visualization
    create_correlation_network(correlations_df, viz_output_dir)
    
    # Get top correlations
    top_positive = correlations_df.nlargest(5, 'Correlation')
    top_negative = correlations_df.nsmallest(5, 'Correlation')
    
    # Prepare correlation results
    correlations_dict = {
        'Full Correlation Matrix': product_corr.round(2),
        'Strongest Positive Correlations': top_positive.to_string(index=False),
        'Strongest Negative Correlations': top_negative.to_string(index=False)
    }
    
    # Print results
    print("\nFull Correlation Matrix:")
    print(product_corr.round(2))
    
    print("\nStrongest Positive Correlations:")
    print(top_positive.to_string(index=False))
    
    print("\nStrongest Negative Correlations:")
    print(top_negative.to_string(index=False))
    
    # Save results
    corr_file = save_correlation_results(correlations_dict, output_dir)
    
    print(f"\nAnalysis results saved to:")
    print(f"- Correlations: {corr_file}")
    print(f"- Visualizations: {viz_output_dir}")

if __name__ == "__main__":
    analyze_correlations()
