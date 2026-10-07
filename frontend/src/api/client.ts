import axios from 'axios';

const apiClient = axios.create({
  baseURL: 'http://localhost:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  checkHealth: () => apiClient.get('/health'),
  getSummary: () => apiClient.get('/data/summary'),
  getCorrelation: () => apiClient.get('/eda/correlation'),
  getDistributions: () => apiClient.get('/eda/distributions'),
  getOutliers: () => apiClient.get('/eda/outliers'),
  evaluateModel: (features: string[], k_min: number, k_max: number) => 
    apiClient.post('/model/evaluate', { features, k_min, k_max }),
  fitModel: (features: string[], k: number) => 
    apiClient.post('/model/fit', { features, k }),
  getSegments: () => apiClient.get('/segments'),
  predict: (unit_price: number, quantity: number, rating: number) =>
    apiClient.post('/predict', { unit_price, quantity, rating })
};
