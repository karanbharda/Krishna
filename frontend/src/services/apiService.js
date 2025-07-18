import axios from 'axios';

// API Base URL - will use proxy in development
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'http://127.0.0.1:5000/api' 
  : '/api';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

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
    
    // Handle specific error cases
    if (error.response?.status === 404) {
      throw new Error('API endpoint not found');
    } else if (error.response?.status === 500) {
      throw new Error('Server error occurred');
    } else if (error.code === 'ECONNABORTED') {
      throw new Error('Request timeout');
    } else if (!error.response) {
      throw new Error('Network error - please check if the backend server is running');
    }
    
    throw error;
  }
);

export const apiService = {
  // Bot Status
  async getStatus() {
    try {
      const response = await api.get('/status');
      return response.data;
    } catch (error) {
      console.error('Error getting bot status:', error);
      throw error;
    }
  },

  // Portfolio Management
  async getPortfolio() {
    try {
      const response = await api.get('/portfolio');
      return response.data;
    } catch (error) {
      console.error('Error getting portfolio:', error);
      throw error;
    }
  },

  // Trading History
  async getTrades(limit = 10) {
    try {
      const response = await api.get(`/trades?limit=${limit}`);
      return response.data;
    } catch (error) {
      console.error('Error getting trades:', error);
      throw error;
    }
  },

  // Watchlist Management
  async getWatchlist() {
    try {
      const response = await api.get('/watchlist');
      return response.data;
    } catch (error) {
      console.error('Error getting watchlist:', error);
      throw error;
    }
  },

  async updateWatchlist(ticker, action) {
    try {
      const response = await api.post('/watchlist', {
        ticker: ticker,
        action: action
      });
      return response.data;
    } catch (error) {
      console.error('Error updating watchlist:', error);
      throw error;
    }
  },

  // Chat/Commands
  async sendChatMessage(message) {
    try {
      const response = await api.post('/chat', {
        message: message
      });
      return response.data;
    } catch (error) {
      console.error('Error sending chat message:', error);
      throw error;
    }
  },

  // Bot Control
  async startBot() {
    try {
      const response = await api.post('/start');
      return response.data;
    } catch (error) {
      console.error('Error starting bot:', error);
      throw error;
    }
  },

  async stopBot() {
    try {
      const response = await api.post('/stop');
      return response.data;
    } catch (error) {
      console.error('Error stopping bot:', error);
      throw error;
    }
  },

  // Settings Management
  async getSettings() {
    try {
      const response = await api.get('/settings');
      return response.data;
    } catch (error) {
      console.error('Error getting settings:', error);
      throw error;
    }
  },

  async updateSettings(settings) {
    try {
      const response = await api.post('/settings', settings);
      return response.data;
    } catch (error) {
      console.error('Error updating settings:', error);
      throw error;
    }
  },

  // Health Check
  async healthCheck() {
    try {
      const response = await api.get('/status');
      return response.status === 200;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }
};

// Utility functions
export const formatCurrency = (amount) => {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    minimumFractionDigits: 2
  }).format(amount);
};

export const formatPercentage = (value) => {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
};

export const formatNumber = (value) => {
  return new Intl.NumberFormat('en-IN').format(value);
};

export default apiService;
