import pandas as pd
from typing import Dict, List, Any

def generate_personas(df: pd.DataFrame, labels: List[int], features: List[str]) -> Dict[int, str]:
    """Generate persona names dynamically based on centroid ranks of Sales, Quantity, or Rating."""
    df_cluster = df.copy()
    df_cluster['Cluster'] = labels
    
    personas = {}
    
    # Calculate means
    means = df_cluster.groupby('Cluster')[features].mean()
    
    if 'Annual Income (k$)' in features and 'Spending Score (1-100)' in features:
        overall_inc = df['Annual Income (k$)'].mean()
        overall_spend = df['Spending Score (1-100)'].mean()
        for c in means.index:
            inc = means.loc[c, 'Annual Income (k$)']
            spend = means.loc[c, 'Spending Score (1-100)']
            if inc > overall_inc * 1.1 and spend > overall_spend * 1.1:
                personas[int(c)] = "VIP / Target"
            elif inc > overall_inc * 1.1 and spend < overall_spend * 0.9:
                personas[int(c)] = "Savers / Careful"
            elif inc < overall_inc * 0.9 and spend > overall_spend * 1.1:
                personas[int(c)] = "Impulse Buyers"
            elif inc < overall_inc * 0.9 and spend < overall_spend * 0.9:
                personas[int(c)] = "Budget / Sensible"
            else:
                personas[int(c)] = "Standard / Average"
    elif 'Sales' in features:
        sales_rank = means['Sales'].rank()
        for c in means.index:
            if sales_rank[c] == sales_rank.max():
                personas[int(c)] = "High Value Spender"
            elif sales_rank[c] == sales_rank.min():
                personas[int(c)] = "Budget Shopper"
            else:
                personas[int(c)] = "Average Shopper"
    elif 'Unit price' in features and 'Quantity' in features:
        # Proxy for sales
        df_cluster['ProxySales'] = df_cluster['Unit price'] * df_cluster['Quantity']
        proxy_means = df_cluster.groupby('Cluster')['ProxySales'].mean()
        sales_rank = proxy_means.rank()
        for c in proxy_means.index:
            if sales_rank[c] == sales_rank.max():
                personas[int(c)] = "High Volume Spender"
            elif sales_rank[c] == sales_rank.min():
                personas[int(c)] = "Discount Hunter"
            else:
                personas[int(c)] = "Routine Shopper"
    else:
        for c in means.index:
            personas[int(c)] = f"Segment {int(c)}"
            
    return personas
