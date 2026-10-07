# Viva Preparation Q&A

**1. Why do we need to standardize data before applying K-Means?**
K-Means uses Euclidean distance. If variables are on vastly different scales (e.g., Sales in hundreds, Rating from 1-10), larger magnitude variables will disproportionately dominate the distance calculation. Standardizing (Z-score) ensures equal feature weighting.

**2. How do the Elbow Method and Silhouette Score validate the choice of 'k'?**
The Elbow Method plots inertia against $k$; the ideal $k$ is at the inflection point where adding clusters yields diminishing returns. The Silhouette Score measures intra-cluster similarity versus inter-cluster distance (-1 to 1). The best $k$ maximizes the silhouette score.

**3. How does K-Means handle categorical variables?**
It does not handle them natively because Euclidean distance is meaningless for discrete categories. Using one-hot encoding creates sparse, high-dimensional spaces that degrade distance metrics. For mixed data, K-Prototypes or K-Modes should be used.

**4. How does K-Means++ improve upon random initialization?**
Standard K-Means randomly picks initial centroids, risking poor local minima. K-Means++ selects the first centroid randomly, then chooses subsequent centroids with probability proportional to their squared distance from existing centroids. This ensures spread-out initialization and faster convergence.

**5. Why use clustering instead of classification for this problem?**
Classification requires labeled training data (e.g., pre-defined categories). In this dataset, there are no target labels. Clustering is an unsupervised exploratory technique used exactly for this scenario: discovering hidden natural groupings in unlabeled data.
