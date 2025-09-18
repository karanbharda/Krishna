import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import PropTypes from 'prop-types';
import { formatCurrency, apiService } from '../services/apiService';

const PortfolioContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 16px;
  flex: 1;
  min-height: 0; /* allow inner containers to size properly */
  overflow: hidden;
`;

const SubTabNav = styled.div`
  display: flex;
  background: white;
  border-radius: 10px;
  padding: 5px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
`;

const SubTabButton = styled.button`
  flex: 1;
  background: ${props => props.$active ? '#3498db' : 'transparent'};
  border: none;
  padding: 12px 16px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.2s ease;
  color: ${props => props.$active ? 'white' : '#7f8c8d'};
  box-shadow: ${props => props.$active ? '0 2px 8px rgba(52, 152, 219, 0.25)' : 'none'};

  &:hover {
    background: ${props => props.$active ? '#2980b9' : '#ecf0f1'};
    color: #2c3e50;
  }
`;

const TabBody = styled.div`
  flex: 1;
  min-height: 0; /* allow child to overflow & scroll */
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const TabPanel = styled.div`
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
  overflow-y: auto; /* full-screen scroll within the tab */
  overflow-x: hidden;
  padding-right: 6px; /* keep space for scrollbar */
`;

const Section = styled.div`
  h3 {
    margin-bottom: 15px;
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 5px;
  }
`;

const HoldingsTable = styled.div`
  background: #f8f9fa;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid #e9ecef;
  max-height: 80vh;
  overflow-y: auto;

  /* Custom scrollbar styling */
  &::-webkit-scrollbar { width: 8px; }
  &::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 4px; }
  &::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 4px; }
  &::-webkit-scrollbar-thumb:hover { background: #a8a8a8; }
`;

const HoldingsHeader = styled.div`
  display: grid;
  grid-template-columns: 1.5fr 0.8fr 1fr 1fr 1fr 1fr 1fr 0.8fr;
  gap: 10px;
  padding: 15px;
  background: linear-gradient(135deg, #3498db, #2980b9);
  color: white;
  font-weight: bold;
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;

  @media (max-width: 768px) {
    grid-template-columns: 1fr 0.8fr 1fr 1fr;
    font-size: 0.7rem;
    gap: 5px;
  }
`;

const HoldingsRow = styled.div`
  display: grid;
  grid-template-columns: 1.5fr 0.8fr 1fr 1fr 1fr 1fr 1fr 0.8fr;
  gap: 10px;
  padding: 15px;
  align-items: center;
  border-bottom: 1px solid #e9ecef;
  background: white;
  transition: all 0.2s ease;

  &:last-child { border-bottom: none; }
  &:hover { background: #f8f9fa; transform: translateY(-1px); box-shadow: 0 2px 4px rgba(0,0,0,0.1); }

  @media (max-width: 768px) {
    grid-template-columns: 1fr 0.8fr 1fr 1fr;
    font-size: 0.8rem;
    gap: 5px;
  }
`;

const TickerName = styled.div`
  font-weight: bold;
`;

const ProfitLoss = styled.div`
  font-weight: bold;
  color: ${props => props.value > 0 ? '#27ae60' : props.value < 0 ? '#e74c3c' : '#7f8c8d'};
`;

const TradeBadge = styled.span`
  display: inline-block;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 0.7rem;
  font-weight: bold;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  background: ${props => props.type === 'BUY' ? '#27ae60' : '#e74c3c'};
  color: white;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
`;

const StatusBadge = styled.span`
  display: inline-block;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 0.7rem;
  font-weight: bold;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: white;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
  background: ${props => {
    switch (props.status) {
      case 'ACTIVE': return '#27ae60';
      case 'PROFIT': return '#2ecc71';
      case 'LOSS': return '#e74c3c';
      default: return '#3498db';
    }
  }};
`;

const LastUpdateTime = styled.div`
  font-size: 0.8rem;
  color: #7f8c8d;
  text-align: right;
  margin-bottom: 10px;
  font-style: italic;
`;

const RealTimeIndicator = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.8rem;
  color: #27ae60;
  margin-bottom: 10px;

  &::before {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #27ae60;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
  }
`;

/* Watchlist UI */
const WatchlistControls = styled.div`
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
  @media(max-width: 768px) { flex-direction: column; }
`;

const TickerInput = styled.input`
  flex: 1;
  padding: 10px;
  border: 2px solid #e9ecef;
  border-radius: 6px;
  font-size: 1rem;
  &:focus { outline: none; border-color: #3498db; }
`;

const AddButton = styled.button`
  background: #27ae60;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.3s ease;
  &:hover { background: #229954; }
  &:disabled { background: #95a5a6; cursor: not-allowed; }
`;

const WatchlistGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 10px;
  @media(max-width: 768px) { grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); }
`;

const WatchlistItem = styled.div`
  background: #3498db;
  color: white;
  padding: 10px;
  border-radius: 6px;
  text-align: center;
  font-weight: 500;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const RemoveButton = styled.button`
  position: absolute;
  top: -5px;
  right: -5px;
  background: #e74c3c;
  color: white;
  border: none;
  border-radius: 50%;
  width: 20px;
  height: 20px;
  cursor: pointer;
  font-size: 0.8rem;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
  &:hover { background: #c0392b; transform: scale(1.1); }
`;

const NoHoldings = styled.div`
  text-align: center;
  color: #7f8c8d;
  font-style: italic;
  padding: 20px;
`;

const CurrentWatchlist = styled.div`
  h4 { color: #2c3e50; margin-bottom: 10px; }
`;

const UploadSection = styled.div`
  margin: 20px 0;
  padding: 15px;
  border: 2px dashed #3498db;
  border-radius: 8px;
  background: #f8f9fa;
  text-align: center;
`;

const UploadButton = styled.label`
  display: inline-block;
  background: #3498db;
  color: white;
  padding: 10px 20px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.3s ease;
  margin: 10px 0;
  &:hover { background: #2980b9; }
  &:disabled { background: #95a5a6; cursor: not-allowed; }
`;

const HiddenFileInput = styled.input`
  display: none;
`;

const UploadStatus = styled.div`
  margin-top: 10px;
  padding: 8px 12px;
  border-radius: 4px;
  font-size: 0.9rem;
  background: ${props =>
    props.status.includes('✅') ? '#d4edda' :
      props.status.includes('❌') ? '#f8d7da' :
        '#e2e3e5'};
  color: ${props =>
    props.status.includes('✅') ? '#155724' :
      props.status.includes('❌') ? '#721c24' :
        '#6c757d'};
  border: 1px solid ${props =>
    props.status.includes('✅') ? '#c3e6cb' :
      props.status.includes('❌') ? '#f5c6cb' :
        '#d1ecf1'};
`;

const UploadInstructions = styled.div`
  font-size: 0.85rem;
  color: #6c757d;
  margin-top: 8px;
  line-height: 1.4;
`;

/* Recent Trading Activity Styles */
const ActivitySection = styled.div`
  background: #f8f9fa;
  padding: 20px;
  border-radius: 10px;
  border: 1px solid #e9ecef;
  display: flex;
  flex-direction: column;
  min-height: 200px;
`;

const ActivityList = styled.div`
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  max-height: 80vh;
  min-height: 120px;
  padding-right: 8px;
  margin-right: -8px;
  &::-webkit-scrollbar { width: 8px; }
  &::-webkit-scrollbar-track { background: #f8f9fa; border-radius: 4px; border: 1px solid #e9ecef; }
  &::-webkit-scrollbar-thumb { background: #6c757d; border-radius: 4px; border: 1px solid #dee2e6; }
  &::-webkit-scrollbar-thumb:hover { background: #495057; }
  scrollbar-width: thin;
  scrollbar-color: #6c757d #f8f9fa;
`;

const ActivityItem = styled.div`
  background: white;
  padding: 16px;
  margin-bottom: 10px;
  border-radius: 8px;
  border-left: 4px solid ${props => props.type === 'BUY' ? '#27ae60' : '#e74c3c'};
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
  min-height: 70px;
`;

const ActivityDetails = styled.div`
  flex: 1;
`;

const ActivityTitle = styled.div`
  font-weight: bold;
  margin-bottom: 5px;
  color: ${props => props.type === 'BUY' ? '#27ae60' : '#e74c3c'};
  font-size: 0.95rem;
`;

const ActivityTime = styled.div`
  font-size: 0.9rem;
  color: #666;
`;

const ActivityValues = styled.div`
  text-align: right;
  font-size: 0.85rem;
  min-width: 120px;
`;

const Portfolio = ({ botData, onAddTicker, onRemoveTicker }) => {
  const [subTab, setSubTab] = useState('holdings'); // 'holdings' | 'activity' | 'watchlist'
  const [newTicker, setNewTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [tradeHistory, setTradeHistory] = useState([]);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [realtimeData, setRealtimeData] = useState(null);
  const [syncLoading, setSyncLoading] = useState(false);
  const [syncStatus, setSyncStatus] = useState('');

  // Fetch trade history on component mount
  useEffect(() => {
    const fetchTradeHistory = async () => {
      try {
        const trades = await apiService.getTrades(50);
        setTradeHistory(trades);
      } catch (error) {
        console.error('Error fetching trade history:', error);
      }
    };
    fetchTradeHistory();
  }, []);

  // Real-time updates every 5 seconds for professional trading
  useEffect(() => {
    const interval = setInterval(async () => {
      setLastUpdate(new Date());
      try {
        // Fetch real-time portfolio data
        const response = await fetch('/api/portfolio/realtime');
        if (response.ok) {
          const data = await response.json();
          setRealtimeData(data);
        }
        // Refetch trade history for real-time updates
        const trades = await apiService.getTrades(50);
        setTradeHistory(trades);
      } catch (error) {
        console.error('Error updating real-time data:', error);
      }
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  // Function to get the most recent trade type for a ticker
  const getLastTradeType = (ticker) => {
    const tickerTrades = tradeHistory.filter(trade => trade.asset === ticker);
    if (tickerTrades.length === 0) return 'BUY';
    const sortedTrades = tickerTrades.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    return (sortedTrades[0].action || '').toUpperCase();
  };

  const handleAddTicker = async () => {
    if (!newTicker.trim()) {
      alert('Please enter a ticker symbol');
      return;
    }
    if (botData.config.tickers.includes(newTicker.toUpperCase())) {
      alert('Ticker already in watchlist');
      return;
    }
    setLoading(true);
    try {
      await onAddTicker(newTicker.toUpperCase());
      setNewTicker('');
    } catch (error) {
      console.error('Error adding ticker:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveTicker = async (ticker) => {
    setLoading(true);
    try {
      await onRemoveTicker(ticker);
    } catch (error) {
      console.error('Error removing ticker:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleAddTicker();
    }
  };

  const handleSyncPortfolio = async () => {
    if (syncLoading) return;
    setSyncLoading(true);
    setSyncStatus('🔄 Syncing with Dhan account...');
    try {
      const response = await fetch('/api/live/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (response.ok) {
        const result = await response.json();
        setSyncStatus(`✅ Synced! Balance: ₹${result.balance || 0}`);
        setLastUpdate(new Date());
        setTimeout(() => setSyncStatus(''), 3000);
      } else {
        const error = await response.json();
        setSyncStatus(`❌ Sync failed: ${error.detail || 'Unknown error'}`);
        setTimeout(() => setSyncStatus(''), 5000);
      }
    } catch (error) {
      console.error('Sync error:', error);
      setSyncStatus('❌ Sync failed: Network error');
      setTimeout(() => setSyncStatus(''), 5000);
    } finally {
      setSyncLoading(false);
    }
  };

  const handleCSVUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setUploadStatus('❌ Please upload a CSV file');
      return;
    }
    setUploadLoading(true);
    setUploadStatus('📤 Processing CSV file...');
    try {
      const text = await file.text();
      const lines = text.split('\n').filter(line => line.trim());
      if (lines.length === 0) {
        setUploadStatus('❌ CSV file is empty');
        setUploadLoading(false);
        return;
      }
      const tickers = [];
      const errors = [];
      lines.forEach((line, index) => {
        const trimmedLine = line.trim();
        if (!trimmedLine) return;
        const columns = trimmedLine.split(',');
        let ticker = columns[0].trim().toUpperCase();
        if (!ticker.includes('.')) { ticker += '.NS'; }
        if (ticker.match(/^[A-Z0-9&-]+\.(NS|BO)$/)) {
          if (!tickers.includes(ticker) && !botData.config.tickers.includes(ticker)) {
            tickers.push(ticker);
          }
        } else {
          errors.push(`Line ${index + 1}: Invalid ticker format "${ticker}"`);
        }
      });
      if (errors.length > 0 && tickers.length === 0) {
        setUploadStatus(`❌ No valid tickers found.Errors: ${errors.slice(0, 3).join(', ')} `);
        setUploadLoading(false);
        return;
      }
      if (tickers.length === 0) {
        setUploadStatus('❌ No new tickers to add (all already in watchlist)');
        setUploadLoading(false);
        return;
      }
      setUploadStatus(`📤 Adding ${tickers.length} tickers...`);
      try {
        const result = await apiService.bulkUpdateWatchlist(tickers, 'ADD');
        setUploadStatus(`✅ ${result.message} `);
        if (result.successful_tickers.length > 0) {
          window.location.reload();
        }
      } catch (error) {
        console.error('Bulk upload failed:', error);
        setUploadStatus('❌ Failed to upload tickers. Please try again.');
      }
      setTimeout(() => setUploadStatus(''), 5000);
    } catch (error) {
      console.error('Error processing CSV:', error);
      setUploadStatus('❌ Error processing CSV file');
    } finally {
      setUploadLoading(false);
      event.target.value = '';
    }
  };

  const calculatePortfolioPercentage = (currentValue, totalValue) => {
    if (!totalValue || totalValue <= 0 || currentValue === null || currentValue === undefined) {
      return '0.0';
    }
    const percentage = ((Math.abs(currentValue) / totalValue) * 100);
    return isNaN(percentage) ? '0.0' : percentage.toFixed(1);
  };

  // Use real-time data if available, otherwise fallback to botData
  const portfolioData = realtimeData?.portfolio_metrics || botData.portfolio;
  const holdings = portfolioData.holdings || {};
  const totalValue = portfolioData.total_value || portfolioData.totalValue || 0;
  const cash = portfolioData.cash || 0;

  return (
    <PortfolioContainer>
      <SubTabNav>
        <SubTabButton $active={subTab === 'holdings'} onClick={() => setSubTab('holdings')}>
          <i className="fas fa-briefcase" style={{ marginRight: 8 }}></i> Current Holding
        </SubTabButton>
        <SubTabButton $active={subTab === 'activity'} onClick={() => setSubTab('activity')}>
          <i className="fas fa-history" style={{ marginRight: 8 }}></i> Recent Trading Activity
        </SubTabButton>
        <SubTabButton $active={subTab === 'watchlist'} onClick={() => setSubTab('watchlist')}>
          <i className="fas fa-list" style={{ marginRight: 8 }}></i> Watchlist
        </SubTabButton>
      </SubTabNav>

      <TabBody>
        {subTab === 'holdings' && (
          <TabPanel>
            <Section>
              <h3>Current Holdings</h3>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                  <RealTimeIndicator>
                    Live Portfolio Data
                  </RealTimeIndicator>
                  {botData?.mode === 'live' && (
                    <button
                      onClick={handleSyncPortfolio}
                      disabled={syncLoading}
                      style={{
                        padding: '8px 16px',
                        backgroundColor: syncLoading ? '#6c757d' : '#007bff',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: syncLoading ? 'not-allowed' : 'pointer',
                        fontSize: '14px',
                        fontWeight: '500'
                      }}
                    >
                      {syncLoading ? '🔄 Syncing...' : '🔄 Sync Now'}
                    </button>
                  )}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                  <LastUpdateTime>
                    Last updated: {lastUpdate.toLocaleTimeString()}
                  </LastUpdateTime>
                  {syncStatus && (
                    <div style={{
                      fontSize: '12px',
                      marginTop: '4px',
                      color: syncStatus.includes('✅') ? '#28a745' : syncStatus.includes('❌') ? '#dc3545' : '#007bff'
                    }}>
                      {syncStatus}
                    </div>
                  )}
                </div>
              </div>

              <HoldingsTable>
                {Object.keys(holdings).length > 0 ? (
                  <>
                    <HoldingsHeader>
                      <div>Stock Symbol</div>
                      <div>Status</div>
                      <div>Last Trade</div>
                      <div>Quantity</div>
                      <div>Avg Price</div>
                      <div>Current Price</div>
                      <div>Profit/Loss</div>
                      <div>% Portfolio</div>
                    </HoldingsHeader>
                    {Object.entries(holdings).length === 0 ? (
                      <HoldingsRow>
                        <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '40px', color: '#7f8c8d' }}>
                          <i className="fas fa-chart-line" style={{ fontSize: '2rem', marginBottom: '10px' }}></i>
                          <div>No stocks in portfolio yet</div>
                          <div style={{ fontSize: '0.9rem', marginTop: '5px' }}>
                            Start the trading bot to begin building your portfolio
                          </div>
                        </div>
                      </HoldingsRow>
                    ) : (
                      Object.entries(holdings).map(([ticker, data]) => {
                        const realtimePrice = realtimeData?.current_prices?.[ticker]?.price;
                        const currentPrice = realtimePrice || data.currentPrice || data.avg_price || 0;
                        const avgPrice = data.avg_price || 0;
                        const qty = data.qty || 0;
                        const currentValue = qty * currentPrice;
                        const costBasis = qty * avgPrice;
                        const profitLoss = currentValue - costBasis;
                        const profitLossPct = costBasis > 0 ? ((profitLoss / costBasis) * 100) : 0;
                        const portfolioPercentage = calculatePortfolioPercentage(currentValue, totalValue);
                        const lastTradeType = getLastTradeType(ticker);
                        const status = qty > 0 ? (profitLoss >= 0 ? 'PROFIT' : 'LOSS') : 'ACTIVE';
                        return (
                          <HoldingsRow key={ticker}>
                            <TickerName>{ticker}</TickerName>
                            <div>
                              <StatusBadge status={status}>{status}</StatusBadge>
                            </div>
                            <div>
                              <TradeBadge type={lastTradeType}>{lastTradeType}</TradeBadge>
                            </div>
                            <div>{qty.toFixed(2)}</div>
                            <div>{formatCurrency(avgPrice)}</div>
                            <div>{formatCurrency(currentPrice)}</div>
                            <ProfitLoss value={profitLoss}>
                              {profitLoss >= 0 ? '+' : ''}{formatCurrency(profitLoss)}
                              <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>
                                ({profitLossPct >= 0 ? '+' : ''}{profitLossPct.toFixed(2)}%)
                              </div>
                            </ProfitLoss>
                            <div>{portfolioPercentage}%</div>
                          </HoldingsRow>
                        );
                      })
                    )}
                  </>
                ) : (
                  <NoHoldings>No current holdings</NoHoldings>
                )}
              </HoldingsTable>
            </Section>
          </TabPanel>
        )}

        {subTab === 'activity' && (
          <TabPanel>
            <Section>
              <h3>Recent Trading Activity</h3>
              <ActivitySection>
                <ActivityList>
                  {tradeHistory && tradeHistory.length > 0 ? (
                    tradeHistory.slice(0, 50).map((trade, index) => {
                      const action = (trade.action || '').toUpperCase();
                      const asset = trade.asset || trade.ticker || 'Unknown';
                      const displayAsset = asset.replace('.NS', '');
                      const qty = trade.qty || trade.quantity || 0;
                      const price = trade.price || 0;
                      const ts = trade.timestamp || new Date().toISOString();
                      const d = new Date(ts);
                      const formatted = `${d.toLocaleDateString('en-IN')} ${d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}`;
                      return (
                        <ActivityItem key={`act-${index}-${asset}-${ts}`} type={action}>
                          <ActivityDetails>
                            <ActivityTitle type={action}>{action} {displayAsset}</ActivityTitle>
                            <ActivityTime>{formatted}</ActivityTime>
                          </ActivityDetails>
                          <ActivityValues>
                            <div><strong>Qty:</strong> {qty}</div>
                            <div><strong>Price:</strong> {formatCurrency(price)}</div>
                            <div><strong>Total:</strong> {formatCurrency(qty * price)}</div>
                          </ActivityValues>
                        </ActivityItem>
                      );
                    })
                  ) : (
                    <div style={{ textAlign: 'center', color: '#7f8c8d', fontStyle: 'italic', padding: '20px', background: 'white', borderRadius: '8px', border: '2px dashed #e9ecef' }}>
                      No recent trades
                    </div>
                  )}
                </ActivityList>
              </ActivitySection>
            </Section>
          </TabPanel>
        )}

        {subTab === 'watchlist' && (
          <TabPanel>
            <Section>
              <h3>Watchlist Management</h3>
              <WatchlistControls>
                <TickerInput
                  type="text"
                  placeholder="Add Ticker (e.g., INFY.NS)"
                  value={newTicker}
                  onChange={(e) => setNewTicker(e.target.value)}
                  onKeyPress={handleKeyPress}
                  disabled={loading}
                />
                <AddButton onClick={handleAddTicker} disabled={loading || !newTicker.trim()}>
                  {loading ? 'Adding...' : 'Add Ticker'}
                </AddButton>
              </WatchlistControls>

              <UploadSection>
                <h4 style={{ margin: '0 0 10px 0', color: '#2c3e50' }}>📁 Bulk Upload Tickers</h4>
                <UploadButton htmlFor="csv-upload" disabled={uploadLoading}>
                  {uploadLoading ? '📤 Processing...' : '📤 Upload CSV File'}
                </UploadButton>
                <HiddenFileInput
                  id="csv-upload"
                  type="file"
                  accept=".csv"
                  onChange={handleCSVUpload}
                  disabled={uploadLoading}
                />
                <UploadInstructions>
                  <strong>CSV Format:</strong> One ticker per line or ticker,name format<br />
                  <strong>Example:</strong> RELIANCE, TCS, HDFCBANK or RELIANCE.NS,Reliance Industries<br />
                  <strong>Note:</strong> .NS suffix will be added automatically for Indian stocks
                </UploadInstructions>
                {uploadStatus && (
                  <UploadStatus status={uploadStatus}>
                    {uploadStatus}
                  </UploadStatus>
                )}
              </UploadSection>

              <CurrentWatchlist>
                <h4>Current Watchlist:</h4>
                <WatchlistGrid>
                  {botData.config.tickers.map(ticker => (
                    <WatchlistItem key={ticker}>
                      {ticker}
                      <RemoveButton
                        onClick={() => handleRemoveTicker(ticker)}
                        disabled={loading}
                        title={`Remove ${ticker} `}
                      >
                        ×
                      </RemoveButton>
                    </WatchlistItem>
                  ))}
                </WatchlistGrid>
              </CurrentWatchlist>
            </Section>
          </TabPanel>
        )}
      </TabBody>
    </PortfolioContainer>
  );
};

Portfolio.propTypes = {
  botData: PropTypes.shape({
    portfolio: PropTypes.object.isRequired,
    config: PropTypes.shape({
      tickers: PropTypes.array.isRequired
    }).isRequired
  }).isRequired,
  onAddTicker: PropTypes.func.isRequired,
  onRemoveTicker: PropTypes.func.isRequired
};

export default Portfolio;
