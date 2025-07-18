import React from 'react';
import styled from 'styled-components';
import { formatCurrency, formatPercentage } from '../services/apiService';

const SidebarContainer = styled.div`
  width: 300px;
  background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
  color: white;
  padding: 20px;
  overflow-y: auto;
`;

const SidebarHeader = styled.div`
  h2 {
    margin-bottom: 20px;
    text-align: center;
    font-size: 1.4rem;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
  }
`;

const ModeIndicator = styled.div`
  margin-bottom: 20px;
`;

const ModeBadge = styled.div`
  padding: 8px 12px;
  border-radius: 20px;
  text-align: center;
  font-weight: bold;
  font-size: 0.9rem;
  background: ${props => props.mode === 'live' ? '#e74c3c' : '#27ae60'};
  color: white;
`;

const SidebarSection = styled.div`
  margin-bottom: 25px;
  
  h3 {
    margin-bottom: 15px;
    color: #3498db;
    font-size: 1.1rem;
    border-left: 3px solid #3498db;
    padding-left: 10px;
  }
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
`;

const MetricCard = styled.div`
  background: rgba(255, 255, 255, 0.1);
  padding: 12px;
  border-radius: 8px;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.2);
`;

const MetricLabel = styled.div`
  font-size: 0.8rem;
  opacity: 0.8;
  margin-bottom: 5px;
`;

const MetricValue = styled.div`
  font-size: 1.2rem;
  font-weight: bold;
  color: #3498db;
`;

const MetricChange = styled.div`
  font-size: 0.9rem;
  margin-top: 5px;
  color: ${props => props.positive ? '#27ae60' : '#e74c3c'};
`;

const QuickActions = styled.div`
  display: flex;
  flex-direction: column;
  gap: 10px;
`;

const ActionButton = styled.button`
  background: #3498db;
  color: white;
  border: none;
  padding: 10px 15px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 8px;

  &:hover {
    background: #2980b9;
    transform: translateY(-2px);
  }

  &:disabled {
    background: #7f8c8d;
    cursor: not-allowed;
    transform: none;
  }

  i {
    font-size: 1rem;
  }
`;

const Sidebar = ({ botData, onStartBot, onPauseBot, onRefresh }) => {
  const calculateMetrics = () => {
    const totalValue = botData.portfolio.totalValue;
    const cash = botData.portfolio.cash;
    const startingBalance = botData.portfolio.startingBalance;
    const totalReturn = totalValue - startingBalance;
    const returnPercentage = (totalReturn / startingBalance) * 100;
    
    return {
      totalValue,
      cash,
      totalReturn,
      returnPercentage
    };
  };

  const metrics = calculateMetrics();
  const positionsCount = Object.keys(botData.portfolio.holdings).length;

  return (
    <SidebarContainer>
      <SidebarHeader>
        <h2>📈 Trading Dashboard</h2>
      </SidebarHeader>

      <ModeIndicator>
        <ModeBadge mode={botData.config.mode}>
          {botData.config.mode === 'live' ? '🔴 LIVE TRADING MODE' : '📝 PAPER TRADING MODE'}
        </ModeBadge>
      </ModeIndicator>

      <SidebarSection>
        <h3>Portfolio Metrics</h3>
        <MetricsGrid>
          <MetricCard>
            <MetricLabel>Total Value</MetricLabel>
            <MetricValue>{formatCurrency(metrics.totalValue)}</MetricValue>
          </MetricCard>
          
          <MetricCard>
            <MetricLabel>Cash</MetricLabel>
            <MetricValue>{formatCurrency(metrics.cash)}</MetricValue>
          </MetricCard>
          
          <MetricCard>
            <MetricLabel>Total Return</MetricLabel>
            <MetricValue>{formatCurrency(metrics.totalReturn)}</MetricValue>
            <MetricChange positive={metrics.returnPercentage >= 0}>
              {formatPercentage(metrics.returnPercentage)}
            </MetricChange>
          </MetricCard>
          
          <MetricCard>
            <MetricLabel>Positions</MetricLabel>
            <MetricValue>{positionsCount}</MetricValue>
          </MetricCard>
        </MetricsGrid>
      </SidebarSection>

      <SidebarSection>
        <h3>Quick Actions</h3>
        <QuickActions>
          {botData.isRunning ? (
            <ActionButton onClick={onPauseBot}>
              <i className="fas fa-pause"></i>
              Pause Bot
            </ActionButton>
          ) : (
            <ActionButton onClick={onStartBot}>
              <i className="fas fa-play"></i>
              Start Bot
            </ActionButton>
          )}
          
          <ActionButton onClick={onRefresh}>
            <i className="fas fa-refresh"></i>
            Refresh
          </ActionButton>
        </QuickActions>
      </SidebarSection>
    </SidebarContainer>
  );
};

export default Sidebar;
