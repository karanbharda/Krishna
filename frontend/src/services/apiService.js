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

  // Complete Bot Data (for React frontend)
  async getBotData() {
    try {
      const response = await api.get('/bot-data');
      return response.data;
    } catch (error) {
      console.error('Error getting bot data:', error);
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

  async bulkUpdateWatchlist(tickers, action = 'ADD') {
    try {
      const response = await api.post('/watchlist/bulk', {
        tickers: tickers,
        action: action
      });
      return response.data;
    } catch (error) {
      console.error('Error bulk updating watchlist:', error);
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

  // ============================================================================
  // MCP (Model Context Protocol) API Endpoints
  // ============================================================================

  // MCP Market Analysis
  async mcpAnalyzeMarket(analysisRequest) {
    try {
      const response = await api.post('/mcp/analyze', analysisRequest);
      return response.data;
    } catch (error) {
      console.error('MCP market analysis error:', error);
      throw error;
    }
  },

  // MCP Trade Execution
  async mcpExecuteTrade(tradeRequest) {
    try {
      const response = await api.post('/mcp/execute', tradeRequest);
      return response.data;
    } catch (error) {
      console.error('MCP trade execution error:', error);
      throw error;
    }
  },

  // MCP Advanced Chat
  async mcpChat(chatRequest) {
    try {
      const response = await api.post('/mcp/chat', chatRequest);
      return response.data;
    } catch (error) {
      console.error('MCP chat error:', error);
      throw error;
    }
  },

  // MCP Status Check
  async getMcpStatus() {
    try {
      const response = await api.get('/mcp/status');
      return response.data;
    } catch (error) {
      console.error('MCP status error:', error);
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

  // Live Trading Status
  async getLiveStatus() {
    try {
      const response = await api.get('/live-status');
      return response.data;
    } catch (error) {
      console.error('Error getting live status:', error);
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
  },

  // Production-level API calls (seamlessly integrated)
  async getSignalPerformance() {
    try {
      const response = await api.get('/production/signal-performance');
      return response.data;
    } catch (error) {
      console.error('Error getting signal performance:', error);
      throw error;
    }
  },

  async getRiskMetrics() {
    try {
      const response = await api.get('/production/risk-metrics');
      return response.data;
    } catch (error) {
      console.error('Error getting risk metrics:', error);
      throw error;
    }
  },

  async makeProductionDecision(symbol) {
    try {
      const response = await api.post('/production/make-decision', { symbol });
      return response.data;
    } catch (error) {
      console.error('Error making production decision:', error);
      throw error;
    }
  },

  async getLearningInsights() {
    try {
      const response = await api.get('/production/learning-insights');
      return response.data;
    } catch (error) {
      console.error('Error getting learning insights:', error);
      throw error;
    }
  },

  async getDecisionHistory(days = 7) {
    try {
      const response = await api.get(`/production/decision-history?days=${days}`);
      return response.data;
    } catch (error) {
      console.error('Error getting decision history:', error);
      throw error;
    }
  }
};

// Utility functions
export const formatCurrency = (amount) => {
  // Handle NaN, undefined, null values
  if (amount === null || amount === undefined || isNaN(amount)) {
    return '₹0.00';
  }

  // Convert to number if it's a string
  const numAmount = typeof amount === 'string' ? parseFloat(amount) : amount;

  // Handle invalid numbers
  if (isNaN(numAmount)) {
    return '₹0.00';
  }

  // For sidebar metrics, use compact notation for large numbers
  if (Math.abs(numAmount) >= 10000000) { // 1 crore
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      notation: 'compact',
      maximumFractionDigits: 2
    }).format(numAmount);
  } else if (Math.abs(numAmount) >= 100000) { // 1 lakh
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0
    }).format(numAmount);
  } else {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 2
    }).format(numAmount);
  }
};

export const formatPercentage = (value) => {
  // Handle NaN, undefined, null values
  if (value === null || value === undefined || isNaN(value)) {
    return '0.00%';
  }

  // Convert to number if it's a string
  const numValue = typeof value === 'string' ? parseFloat(value) : value;

  // Handle invalid numbers
  if (isNaN(numValue)) {
    return '0.00%';
  }

  return `${numValue >= 0 ? '+' : ''}${numValue.toFixed(2)}%`;
};

export const formatNumber = (value) => {
  return new Intl.NumberFormat('en-IN').format(value);
};

export default apiService;
