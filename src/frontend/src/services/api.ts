import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes for large file uploads
});

export interface HealthResponse {
  status: string;
  timestamp: string;
  components: {
    model_loaded: boolean;
    feature_extractor: boolean;
    preprocessor: boolean;
    explainer: boolean;
  };
}

export interface StatsResponse {
  uploaded_files: number;
  total_upload_size: number;
  recent_uploads: Array<{
    filename: string;
    size: number;
    upload_time: string;
  }>;
  system_info: {
    torch_version: string;
    device_available: boolean;
    device_count: number;
  };
}

export interface ModelInfoResponse {
  model_type: string;
  total_parameters: number;
  trainable_parameters: number;
  device: string;
  input_dim: number | string;
  num_classes: number | string;
  model_loaded: boolean;
}

export interface UploadResponse {
  message: string;
  filename: string;
  filepath: string;
  file_size: number;
}

export interface AnalysisResponse {
  filename: string;
  features: Record<string, any>;
  feature_count: number;
  prediction?: {
    risk_level: string;
    confidence: number;
    probabilities: {
      Low: number;
      Medium: number;
      High: number;
    };
    feature_importance: number[];
    top_features: any[];
    explanation?: {
      risk_level: string;
      confidence: number;
      probabilities: {
        Low: number;
        Medium: number;
        High: number;
      };
      top_features: Array<{
        name: string;
        importance: number;
        description: string;
        value: any;
      }>;
      explanations: {
        rule_based: string;
        ai_generated: string | null;
        primary: string;
      };
      risk_factors: {
        permissions: string[];
        apis: string[];
        behavior: string[];
        certificate: string[];
      };
      recommendations: string[];
      timestamp: string;
    };
  };
  timestamp: string;
}

export interface ConfigResponse {
  feature_extraction: Record<string, any>;
  model: Record<string, any>;
  api: Record<string, any>;
  explainability: Record<string, any>;
}

export const apiService = {
  // Health check
  getHealth: async (): Promise<HealthResponse> => {
    const response = await api.get('/health');
    return response.data;
  },

  // Statistics
  getStats: async (): Promise<StatsResponse> => {
    const response = await api.get('/stats');
    return response.data;
  },

  // Model information
  getModelInfo: async (): Promise<ModelInfoResponse> => {
    const response = await api.get('/model/info');
    return response.data;
  },

  // Configuration
  getConfig: async (): Promise<ConfigResponse> => {
    const response = await api.get('/config');
    return response.data;
  },

  // Upload APK
  uploadAPK: async (formData: FormData): Promise<UploadResponse> => {
    const response = await api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Analyze APK
  analyzeAPK: async (data: { filepath: string }): Promise<AnalysisResponse> => {
    const response = await api.post('/analyze', data);
    return response.data;
  },

  // Batch analyze
  batchAnalyze: async (data: { filepaths: string[] }): Promise<any> => {
    const response = await api.post('/batch_analyze', data);
    return response.data;
  },
};

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export default api;