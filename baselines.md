## Baselines

### Baseline Details

- **OCSVM**
  A classical anomaly detector that learns a decision boundary around normal data in feature space. It is effective in low-dimensional settings but struggles with complex temporal dependencies.

- **Isolation Forest (IF)**
  A tree-based ensemble method that isolates anomalies through recursive data partitioning. It assumes that anomalies are easier to isolate than normal points.

- **Local Outlier Factor (LOF)**
  A density-based method that measures local deviation of a given data point with respect to its neighbors. Lower local density implies higher likelihood of being an outlier.

- **Series2Graph (S2G)**
  Transforms time series into graph representations to capture latent correlations. Anomalies are detected based on deviations in graph connectivity patterns, enabling multi-scale dependency analysis.

- **SAND**
  A subsequence anomaly detection framework for streaming time series, which incrementally clusters subsequences and updates centroids using shape-based distances. It adapts to distribution drifts by maintaining a weighted summary of recurrent patterns without storing raw data.

- **TranAD**
  A transformer-based anomaly detection model that combines dual reconstruction paths with adversarial training to learn robust representations of normal behavior. Anomalies are detected based on reconstruction errors and discriminator feedback on prediction residuals.

- **Autoformer** 
  A transformer-based forecasting model that integrates series decomposition and an auto-correlation mechanism to capture long-term trends and seasonal dependencies.

- **Informer**
  A transformer variant designed for long-sequence time series forecasting, introducing the ProbSparse self-attention mechanism to reduce time and memory complexity.

- **FEDformer** 
  A transformer-based forecasting model that enhances long-term prediction by applying Fourier transform to extract frequency-domain features and combining them with decomposition-based temporal modeling.

- **iTransformer** 
  A forecasting-oriented model that inverts the conventional transformer structure by treating temporal and channel dimensions symmetrically, enabling independent channel modeling and efficient extraction of frequency-aware representations for multivariate time series.

- **PatchTST** 
  A channel-independent transformer model for time series forecasting that tokenizes each variable into non-overlapping patches and learns temporal dependencies using 1D patch attention, enabling efficient long-term prediction without inter-channel interference.

- **Dlinear** 
  A decomposition-based linear forecasting model that separately projects trend and seasonal components using linear layers, achieving strong performance with minimal model complexity and computation.

- **AnomalyTransformer (AnoTrans)** 
  A transformer-based model that introduces a prior-association mechanism to measure association discrepancy between normal and anomalous patterns, enabling both accurate detection and interpretable identification of anomalies in time series.

- **DCDetector** 
  A streaming anomaly detection framework that combines multi-scale convolutional encoders with graph attention to disentangle temporal and contextual features, enabling real-time detection on multivariate time series.

- **UniTS**
  A universal anomaly detection framework that unifies spatial-temporal modeling across diverse time series by integrating adaptive decompositions, dynamic thresholds, and hierarchical temporal encoding for robust and generalizable detection.
