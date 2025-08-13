import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import toast from 'react-hot-toast';

// Components
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import Portfolio from './components/Portfolio';
import ChatAssistant from './components/ChatAssistant';
import LoadingOverlay from './components/LoadingOverlay';
import SettingsModal from './components/SettingsModal';

// Services
import { apiService } from './services/apiService';

// Styled Components
const AppContainer = styled.div`
  display: flex;
  min-height: 100vh;
  width: 100%;
  background: #ffffff;
`;

const MainContent = styled.div`
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  overflow-x: hidden;
  display: flex;
  flex-direction: column;
  min-height: 100vh;

  /* Custom scrollbar for main content */
  &::-webkit-scrollbar {
    width: 8px;
  }

  &::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.05);
    border-radius: 4px;
  }

  &::-webkit-scrollbar-thumb {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
  }

  &::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 0, 0, 0.3);
  }

  /* Firefox scrollbar */
  scrollbar-width: thin;
  scrollbar-color: rgba(0, 0, 0, 0.2) rgba(0, 0, 0, 0.05);
`;

const TabContent = styled.div`
  background: white;
  border-radius: 12px;
  padding: 25px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  flex: 1;
  overflow: visible;
  display: flex;
  flex-direction: column;
  margin-bottom: 20px;
`;

function App() {
  // State Management
  const [activeTab, setActiveTab] = useState('dashboard');
  const [botData, setBotData] = useState({
    portfolio: {
      totalValue: 1000000,
      cash: 1000000,
      holdings: {},
      tradeLog: [],
      startingBalance: 1000000
    },
    config: {
      mode: 'paper',
      tickers: [], // Empty by default - users can add tickers manually
      riskLevel: 'MEDIUM',
      maxAllocation: 25
    },
    isRunning: false,
    chatMessages: []
  });

  const [loading, setLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [liveStatus, setLiveStatus] = useState(null);
  const [mcpAvailable, setMcpAvailable] = useState(false);

  // Initialize app and load data
  useEffect(() => {
    initializeApp();
    const interval = setInterval(refreshData, 30000); // Auto-refresh every 30 seconds
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const initializeApp = async () => {
    try {
      setLoading(true);
      await loadDataFromBackend();
      await loadLiveStatus();
      await checkMcpStatus();

      // Add welcome message if no messages exist
      if (botData.chatMessages.length === 0) {
        setBotData(prev => ({
          ...prev,
          chatMessages: [{
            role: 'assistant',
            content: 'Welcome to the Indian Stock Trading Bot! 🚀\nType a command or ask me anything about trading and markets.',
            timestamp: new Date().toISOString()
          }]
        }));
      }
    } catch (error) {
      console.error('Error initializing app:', error);
      toast.error('Failed to initialize application');
    } finally {
      setLoading(false);
    }
  };

  const loadDataFromBackend = async () => {
    try {
      // Get complete bot data from new endpoint
      const botData = await apiService.getBotData();

      setBotData(prev => ({
        ...prev,
        ...botData,
        chatMessages: prev.chatMessages // Preserve chat messages
      }));

    } catch (error) {
      console.error('Error loading data from backend:', error);
      // Fall back to localStorage if available
      const savedData = localStorage.getItem('tradingBotData');
      if (savedData) {
        try {
          const parsed = JSON.parse(savedData);
          setBotData(prev => ({ ...prev, ...parsed }));
        } catch (e) {
          console.error('Error loading saved data:', e);
        }
      }
    }
  };

  const refreshData = async () => {
    try {
      await loadDataFromBackend();
      await loadLiveStatus();
      await checkMcpStatus();
    } catch (error) {
      console.error('Error refreshing data:', error);
    }
  };

  const loadLiveStatus = async () => {
    try {
      const status = await apiService.getLiveStatus();
      setLiveStatus(status);
    } catch (error) {
      console.error('Error loading live status:', error);
      setLiveStatus(null);
    }
  };

  const checkMcpStatus = async () => {
    try {
      const status = await apiService.getMcpStatus();
      setMcpAvailable(Boolean(status?.mcp_available && status?.server_initialized !== false));
    } catch (error) {
      setMcpAvailable(false);
    }
  };

  const startBot = async () => {
    try {
      setLoading(true);
      await apiService.startBot();
      setBotData(prev => ({ ...prev, isRunning: true }));
      toast.success('Trading bot started successfully!');
      await refreshData();
    } catch (error) {
      console.error('Error starting bot:', error);
      toast.error('Failed to start bot');
    } finally {
      setLoading(false);
    }
  };

  const stopBot = async () => {
    try {
      setLoading(true);
      await apiService.stopBot();
      setBotData(prev => ({ ...prev, isRunning: false }));
      toast.success('Trading bot stopped');
      await refreshData();
    } catch (error) {
      console.error('Error stopping bot:', error);
      toast.error('Failed to stop bot');
    } finally {
      setLoading(false);
    }
  };

  const addTicker = async (ticker) => {
    try {
      setLoading(true);
      const response = await apiService.updateWatchlist(ticker, 'ADD');
      setBotData(prev => ({
        ...prev,
        config: { ...prev.config, tickers: response.tickers }
      }));
      toast.success(response.message);
    } catch (error) {
      console.error('Error adding ticker:', error);
      toast.error(`Failed to add ${ticker} to watchlist`);
    } finally {
      setLoading(false);
    }
  };

  const removeTicker = async (ticker) => {
    try {
      setLoading(true);
      const response = await apiService.updateWatchlist(ticker, 'REMOVE');
      setBotData(prev => ({
        ...prev,
        config: { ...prev.config, tickers: response.tickers }
      }));
      toast.success(response.message);
    } catch (error) {
      console.error('Error removing ticker:', error);
      toast.error(`Failed to remove ${ticker} from watchlist`);
    } finally {
      setLoading(false);
    }
  };

  const sendChatMessage = async (message) => {
    try {
      // Add user message immediately
      setBotData(prev => ({
        ...prev,
        chatMessages: [...prev.chatMessages, {
          role: 'user',
          content: message,
          timestamp: new Date().toISOString()
        }]
      }));

      setLoading(true);
      let response;

      // MCP-aware routing
      const lower = message.toLowerCase();
      const isMarketQuery = lower.includes('analyze') || lower.includes('stock') || lower.includes('price');
      const analyzeCmd = lower.startsWith('/analyze');

      if (mcpAvailable && (isMarketQuery || analyzeCmd)) {
        if (analyzeCmd) {
          const parts = message.split(/\s+/);
          const symbol = parts[1] || 'NSE:RELIANCE-EQ';
          try {
            const analysis = await apiService.mcpAnalyzeMarket({
              symbol,
              timeframe: '1D',
              analysis_type: 'comprehensive'
            });
            response = {
              response: `Recommendation: ${analysis.recommendation}\nConfidence: ${(analysis.confidence * 100).toFixed(2)}%\nCurrent: ₹${analysis.current_price?.toFixed?.(2) ?? analysis.current_price}\nTarget: ₹${analysis.target_price?.toFixed?.(2) ?? analysis.target_price}\nStop Loss: ₹${analysis.stop_loss?.toFixed?.(2) ?? analysis.stop_loss}\nReasoning: ${analysis.reasoning || 'N/A'}`,
              timestamp: new Date().toISOString()
            };
          } catch (err) {
            // Fall back to chat if analysis fails
            response = await apiService.mcpChat({ message, context: { type: 'market_analysis' } });
          }
        } else {
          response = await apiService.mcpChat({ message, context: { type: 'market_analysis' } });
        }
      } else {
        response = await apiService.sendChatMessage(message);
      }

      // Add bot response
      setBotData(prev => ({
        ...prev,
        chatMessages: [...prev.chatMessages, {
          role: 'assistant',
          content: response.response,
          timestamp: response.timestamp
        }]
      }));

      // Refresh data if it was a command
      if (message.startsWith('/')) {
        await refreshData();
      }
    } catch (error) {
      console.error('Error sending chat message:', error);
      setBotData(prev => ({
        ...prev,
        chatMessages: [...prev.chatMessages, {
          role: 'assistant',
          content: 'Sorry, I encountered an error processing your message. Please try again.',
          timestamp: new Date().toISOString()
        }]
      }));
    } finally {
      setLoading(false);
    }
  };

  const saveSettings = async (settings) => {
    try {
      setLoading(true);
      await apiService.updateSettings(settings);

      // Refresh data to get the actual current state (in case mode switch failed)
      await refreshData();

      toast.success('Settings saved successfully!');
      setShowSettings(false);
    } catch (error) {
      console.error('Error saving settings:', error);
      toast.error('Failed to save settings');

      // Still refresh data to get current state even if save failed
      try {
        await refreshData();
      } catch (refreshError) {
        console.error('Error refreshing data after failed save:', refreshError);
      }
    } finally {
      setLoading(false);
    }
  };

  // Save data to localStorage
  useEffect(() => {
    localStorage.setItem('tradingBotData', JSON.stringify(botData));
  }, [botData]);

  const renderTabContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard botData={botData} />;
      case 'portfolio':
        return (
          <Portfolio
            botData={botData}
            onAddTicker={addTicker}
            onRemoveTicker={removeTicker}
          />
        );
      case 'chat':
        return (
          <ChatAssistant
            messages={botData.chatMessages}
            onSendMessage={sendChatMessage}
          />
        );
      default:
        return <Dashboard botData={botData} />;
    }
  };

  return (
    <Router>
      <AppContainer>
        <Sidebar
          botData={botData}
          onStartBot={startBot}
          onStopBot={stopBot}
          onRefresh={refreshData}
        />

        <MainContent>
          <Header
            botData={botData}
            activeTab={activeTab}
            onTabChange={setActiveTab}
            onOpenSettings={() => setShowSettings(true)}
            liveStatus={liveStatus}
          />

          <TabContent>
            <Routes>
              <Route path="/" element={renderTabContent()} />
              <Route path="/dashboard" element={<Dashboard botData={botData} />} />
              <Route path="/portfolio" element={
                <Portfolio
                  botData={botData}
                  onAddTicker={addTicker}
                  onRemoveTicker={removeTicker}
                />
              } />
              <Route path="/chat" element={<ChatAssistant messages={botData.chatMessages} onSendMessage={sendChatMessage} />} />
            </Routes>
          </TabContent>
        </MainContent>

        {loading && <LoadingOverlay />}

        {showSettings && (
          <SettingsModal
            settings={botData.config}
            onSave={saveSettings}
            onClose={() => setShowSettings(false)}
          />
        )}
      </AppContainer>
    </Router>
  );
}

export default App;
