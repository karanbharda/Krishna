import React from 'react';
import styled from 'styled-components';
import PropTypes from 'prop-types';
import { formatCurrency, formatPercentage } from '../services/apiService';

const SidebarContainer = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 320px;
  max-width: 85vw;
  height: 100vh;
  color: #ffffff;
  padding: 20px 18px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  z-index: 9998;

  /* Glassmorphism */
  background: rgba(17, 25, 40, 0.65);
  border-right: 1px solid rgba(255, 255, 255, 0.18);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
  backdrop-filter: blur(14px) saturate(160%);
  -webkit-backdrop-filter: blur(14px) saturate(160%);

  /* Slide in/out */
  transform: translateX(${props => (props.$open ? '0' : '-105%')});
  transition: transform 0.35s cubic-bezier(0.22, 1, 0.36, 1);

  @media (max-width: 768px) {
    width: 85vw;
  }
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
  margin-bottom: 30px;

  h3 {
    margin-bottom: 18px;
    color: #3498db;
    font-size: 1.1rem;
    font-weight: 600;
    border-left: 4px solid #3498db;
    padding-left: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 10px;

  @media (max-width: 1200px) {
    grid-template-columns: 1fr;
    gap: 10px;
  }

  @media (max-width: 768px) {
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
`;

const MetricCard = styled.div`
  background: rgba(255, 255, 255, 0.1);
  padding: 15px 12px;
  border-radius: 10px;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.2);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  min-height: 85px;
  transition: all 0.3s ease;

  &:hover {
    background: rgba(255, 255, 255, 0.15);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  }
`;

const MetricLabel = styled.div`
  font-size: 0.75rem;
  opacity: 0.9;
  margin-bottom: 6px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #bdc3c7;
`;

const MetricValue = styled.div`
  font-size: 1.1rem;
  font-weight: 700;
  color: #ffffff;
  margin-bottom: 2px;
  line-height: 1.2;
  text-align: center;
  word-break: break-all;
`;

const MetricChange = styled.div`
  font-size: 0.8rem;
  margin-top: 3px;
  font-weight: 600;
  color: ${props => props.$positive ? '#2ecc71' : '#e74c3c'};
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 2px;

  &::before {
    content: '${props => props.$positive ? '↗' : '↘'}';
    font-size: 0.9rem;
  }
`;

const QuickActions = styled.div`
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: auto;
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

const StartButton = styled(ActionButton)`
  background: #27ae60;

  &:hover:not(:disabled) {
    background: #229954;
  }
`;

const StopButton = styled(ActionButton)`
  background: #e74c3c;

  &:hover:not(:disabled) {
    background: #c0392b;
  }
`;

const RefreshButton = styled(ActionButton)`
  background: #f39c12;

  &:hover:not(:disabled) {
    background: #e67e22;
  }
`;

const ToggleButton = styled.button`
  position: fixed;
  left: 16px;
  bottom: 16px;
  width: 52px;
  height: 52px;
  border-radius: 50%;
  border: 1px solid rgba(255, 255, 255, 0.25);
  background: rgba(17, 25, 40, 0.55);
  color: #ecf0f1;
  z-index: 9999;
  cursor: pointer;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
  backdrop-filter: blur(10px) saturate(160%);
  -webkit-backdrop-filter: blur(10px) saturate(160%);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: transform 0.2s ease, background 0.2s ease;

  &:hover {
    transform: translateY(-2px);
    background: rgba(17, 25, 40, 0.7);
  }

  i {
    font-size: 1.2rem;
  }
`;

const Sidebar = ({ botData, onStartBot, onStopBot, onRefresh }) => {
  const [isOpen, setIsOpen] = React.useState(false);
  const calculateMetrics = () => {
    const totalValue = botData.portfolio.totalValue;
    const cash = botData.portfolio.cash;
    const startingBalance = botData.portfolio.startingBalance;

    // Calculate cash invested (starting balance minus current cash)
    const cashInvested = startingBalance - cash;

    // Calculate total return based on cash invested, not total portfolio value
    // Total return = (current holdings value - cash invested)
    const holdingsValue = totalValue - cash;
    const totalReturn = holdingsValue - cashInvested;

    // Calculate return percentage based on cash invested (avoid division by zero)
    const returnPercentage = cashInvested > 0 ? (totalReturn / cashInvested) * 100 : 0;

    return {
      totalValue,
      cash,
      totalReturn,
      returnPercentage,
      cashInvested
    };
  };

  const metrics = calculateMetrics();
  const positionsCount = Object.keys(botData.portfolio.holdings).length;

  return (
    <>
      <ToggleButton onClick={() => setIsOpen(prev => !prev)} aria-label={isOpen ? 'Close sidebar' : 'Open sidebar'}>
        <i className={`fas ${isOpen ? 'fa-times' : 'fa-bars'}`}></i>
      </ToggleButton>

      <SidebarContainer $open={isOpen}>
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
            <MetricChange $positive={metrics.returnPercentage >= 0}>
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
          <StartButton onClick={onStartBot} disabled={botData.isRunning}>
            <i className="fas fa-play"></i>
            Start Bot
          </StartButton>

          <StopButton onClick={onStopBot} disabled={!botData.isRunning}>
            <i className="fas fa-stop"></i>
            Stop Bot
          </StopButton>

          <RefreshButton onClick={onRefresh}>
            <i className="fas fa-refresh"></i>
            Refresh Data
          </RefreshButton>
        </QuickActions>
      </SidebarSection>
    </SidebarContainer>
    </>
  );
};

Sidebar.propTypes = {
  botData: PropTypes.shape({
    portfolio: PropTypes.object.isRequired,
    config: PropTypes.object.isRequired,
    isRunning: PropTypes.bool.isRequired
  }).isRequired,
  onStartBot: PropTypes.func.isRequired,
  onStopBot: PropTypes.func.isRequired,
  onRefresh: PropTypes.func.isRequired
};

export default Sidebar;
