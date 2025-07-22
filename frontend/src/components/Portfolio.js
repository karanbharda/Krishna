import React, { useState } from 'react';
import styled from 'styled-components';
import { formatCurrency } from '../services/apiService';

const PortfolioContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
  height: 100%;
  overflow: hidden;
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
`;

const HoldingsHeader = styled.div`
  display: grid;
  grid-template-columns: 2fr 1fr 1fr 1fr 1fr;
  gap: 15px;
  padding: 15px;
  background: #3498db;
  color: white;
  font-weight: bold;

  @media (max-width: 768px) {
    grid-template-columns: 1fr 1fr 1fr;
    font-size: 0.9rem;
  }
`;

const HoldingsRow = styled.div`
  display: grid;
  grid-template-columns: 2fr 1fr 1fr 1fr 1fr;
  gap: 15px;
  padding: 15px;
  align-items: center;
  border-bottom: 1px solid #e9ecef;
  background: white;

  &:last-child {
    border-bottom: none;
  }

  &:hover {
    background: #f1f3f4;
  }

  @media (max-width: 768px) {
    grid-template-columns: 1fr 1fr 1fr;
    font-size: 0.9rem;
  }
`;

const TickerName = styled.div`
  font-weight: bold;
`;

const WatchlistControls = styled.div`
  display: flex;
  gap: 10px;
  margin-bottom: 20px;

  @media (max-width: 768px) {
    flex-direction: column;
  }
`;

const TickerInput = styled.input`
  flex: 1;
  padding: 10px;
  border: 2px solid #e9ecef;
  border-radius: 6px;
  font-size: 1rem;

  &:focus {
    outline: none;
    border-color: #3498db;
  }
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

  &:hover {
    background: #229954;
  }

  &:disabled {
    background: #95a5a6;
    cursor: not-allowed;
  }
`;

const WatchlistGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 10px;

  @media (max-width: 768px) {
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  }
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

  &:hover {
    background: #c0392b;
    transform: scale(1.1);
  }
`;

const NoHoldings = styled.div`
  text-align: center;
  color: #7f8c8d;
  font-style: italic;
  padding: 20px;
`;

const CurrentWatchlist = styled.div`
  h4 {
    color: #2c3e50;
    margin-bottom: 10px;
  }
`;

const Portfolio = ({ botData, onAddTicker, onRemoveTicker }) => {
  const [newTicker, setNewTicker] = useState('');
  const [loading, setLoading] = useState(false);

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

  const calculatePortfolioPercentage = (currentValue, totalValue) => {
    if (!totalValue || totalValue <= 0 || !currentValue || currentValue <= 0) {
      return '0.0';
    }
    const percentage = ((currentValue / totalValue) * 100);
    return isNaN(percentage) ? '0.0' : percentage.toFixed(1);
  };

  const holdings = botData.portfolio.holdings || {};
  const totalValue = botData.portfolio.totalValue || 0;

  return (
    <PortfolioContainer>
      {/* Current Holdings */}
      <Section>
        <h3>Current Holdings</h3>
        <HoldingsTable>
          {Object.keys(holdings).length > 0 ? (
            <>
              <HoldingsHeader>
                <div>Ticker</div>
                <div>Quantity</div>
                <div>Avg Price</div>
                <div>Current Value</div>
                <div>% of Portfolio</div>
              </HoldingsHeader>
              {Object.entries(holdings).map(([ticker, data]) => {
                // Use currentPrice if available, otherwise fall back to avgPrice
                const currentPrice = data.currentPrice || data.avgPrice || 0;
                const avgPrice = data.avgPrice || 0;
                const qty = data.qty || 0;
                const currentValue = qty * currentPrice;
                const portfolioPercentage = calculatePortfolioPercentage(currentValue, totalValue);

                return (
                  <HoldingsRow key={ticker}>
                    <TickerName>{ticker}</TickerName>
                    <div>{qty}</div>
                    <div>{formatCurrency(avgPrice)}</div>
                    <div>{formatCurrency(currentValue)}</div>
                    <div>{portfolioPercentage}%</div>
                  </HoldingsRow>
                );
              })}
            </>
          ) : (
            <NoHoldings>No current holdings</NoHoldings>
          )}
        </HoldingsTable>
      </Section>

      {/* Watchlist Management */}
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
          <AddButton
            onClick={handleAddTicker}
            disabled={loading || !newTicker.trim()}
          >
            {loading ? 'Adding...' : 'Add Ticker'}
          </AddButton>
        </WatchlistControls>

        <CurrentWatchlist>
          <h4>Current Watchlist:</h4>
          <WatchlistGrid>
            {botData.config.tickers.map(ticker => (
              <WatchlistItem key={ticker}>
                {ticker}
                <RemoveButton
                  onClick={() => handleRemoveTicker(ticker)}
                  disabled={loading}
                  title={`Remove ${ticker}`}
                >
                  ×
                </RemoveButton>
              </WatchlistItem>
            ))}
          </WatchlistGrid>
        </CurrentWatchlist>
      </Section>
    </PortfolioContainer>
  );
};

export default Portfolio;
