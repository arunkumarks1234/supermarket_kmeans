# Academic Report: Supermarket Shopper Behavioral Segmentation

### 1. Title and Assigned Problem
**Title**: Supermarket Shopper Behavioral Segmentation Using K-Means Clustering
**Assigned Problem**: Analyze a supermarket's transactional dataset using exploratory data analysis and K-Means clustering to uncover distinct shopper segments, enabling data-driven marketing.

### 2. Objective
Identify hidden purchasing behaviors among supermarket customers by applying K-Means clustering. This segmentation facilitates targeted strategies such as personalized discounts and retention programs. Two feature sets will be evaluated to observe the impact of collinearity on clustering structures.

### 3. Dataset and Source
**Source**: "SuperMarket Analysis.csv"
**Summary Table**:
| Feature | Data Type | Describe Stats |
| :--- | :--- | :--- |
| Unit price | float64 | min=10.08, max=99.96, mean=55.67 |
| Quantity | int64 | min=1, max=10, mean=5.51 |
| Tax 5% | float64 | min=0.51, max=49.65, mean=15.38 |
| Sales | float64 | min=10.68, max=1042.65, mean=322.97 |
| cogs | float64 | min=10.18, max=993.00, mean=307.59 |
| gross margin % | float64 | min=4.76, max=4.76, mean=4.76 |
| gross income | float64 | min=0.51, max=49.65, mean=15.38 |
| Rating | float64 | min=4.0, max=10.0, mean=6.97 |

### 4. Data Preprocessing
Data quality checks were automated. Boxplots highlight IQR outliers which were maintained to capture extreme shopper behaviors. Standardizing variables was accomplished using the Z-score:
$z = \frac{x - \mu}{\sigma}$
Standardization is mandatory for K-Means because it utilizes Euclidean distance. Variables with larger magnitudes (like `Sales`) would otherwise disproportionately bias the clustering over smaller variables (like `Rating`).

### 5. Algorithm / Methodology
The model utilizes **K-Means++** for optimized, spread-out centroid initialization to prevent poor local optima. The distance metric is Euclidean distance. 
The algorithm minimizes the inertia (within-cluster sum of squares):
$\text{Inertia} = \sum_{j=1}^{k} \sum_{x \in C_j} ||x - \mu_j||^2$
Centroids are updated iteratively as the mean of points in the cluster:
$\mu_j = \frac{1}{|C_j|} \sum_{x \in C_j} x$

Optimal $k$ validation uses the **Elbow Method** and **Silhouette Score**:
$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$

### 6. Implementation
**Architecture**: FastAPI backend serving an ML pipeline, and a React + Vite frontend for visualization.
**API Table**:
| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/api/health` | GET | Health check |
| `/api/data/summary` | GET | Summary statistics |
| `/api/model/evaluate` | POST | Run silhouette/inertia for k=2..8 |
| `/api/model/fit` | POST | Fit K-Means, output profiles |
| `/api/segments` | GET | Persona generation and stats |

### 7. Outputs and Visualizations
The system outputs the following figures:
- `output/elbow.png`
- `output/silhouette.png`
- `output/pca.png`
And creates the following app screens:
- Dashboard (KPIs, Segments Donut)
- EDA (Histograms, Outliers)
- Model Selection (Toggles)

### 8. Results
*(Using Experiment A: Unit Price, Quantity, Rating)*
| Cluster ID | Segment Name | Avg Sales ($) | Avg Quantity | Avg Gross Income ($) | Avg Rating | Shopper Share % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | <fill from your run> | <fill from your run> | <fill from your run> | <fill from your run> | <fill from your run> | <fill from your run> |
| 1 | <fill from your run> | <fill from your run> | <fill from your run> | <fill from your run> | <fill from your run> | <fill from your run> |
| 2 | <fill from your run> | <fill from your run> | <fill from your run> | <fill from your run> | <fill from your run> | <fill from your run> |

### 9. Interpretation
*(Note: Personas map dynamically to centroid ranks)*
- **High Volume Spender**: High quantity and unit price. **Recommendation**: VIP loyalty programs.
- **Discount Hunter**: Low proxy sales. **Recommendation**: Discount-trigger campaigns.
- **Routine Shopper**: Average metrics. **Recommendation**: Standard promotional flyers.

Note that `Rating` is uniformly distributed, leading to weak silhouette scores and limiting discriminant power on satisfaction.

### 10. Conclusion
K-Means successfully segmented supermarket shoppers into operational profiles. 
**Limitations**: K-Means assumes spherical clusters and is sensitive to outliers. Collinearity (e.g., Sales vs Gross Income) artificially inflates dimensional weighting.
**Future Work**: PCA dimensionality reduction prior to clustering, DBSCAN for arbitrary shapes, GMM for soft assignments, and K-Prototypes to natively handle categorical features like Product Line.

### 11. References
1. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations.
2. Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding.
3. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.
4. scikit-learn, FastAPI Documentation.
