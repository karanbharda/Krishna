import React, { useState } from 'react';
import styled, { keyframes } from 'styled-components';
import PropTypes from 'prop-types';

const pulse = keyframes`
  0% { opacity: 1; }
  50% { opacity: 0.5; }
  100% { opacity: 1; }
`;

const HeaderContainer = styled.header`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 15px;
  border-bottom: 2px solid #ecf0f1;
`;

const Title = styled.h1`
  color: #2c3e50;
  font-size: 2rem;
  margin: 0;
`;

const HeaderControls = styled.div`
  display: flex;
  align-items: center;
  gap: 15px;
`;

const ModeIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 20px;
  font-size: 0.9rem;
  font-weight: 600;
  background: ${props => props.mode === 'live' ? '#e8f5e8' : '#f0f8ff'};
  color: ${props => props.mode === 'live' ? '#2e7d32' : '#1976d2'};
  border: 1px solid ${props => props.mode === 'live' ? '#4caf50' : '#2196f3'};
`;

const StatusDot = styled.div`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${props => {
    if (props.mode === 'live') {
      return props.$connected ? '#4caf50' : '#f44336';
    }
    return '#2196f3';
  }};
  animation: ${props => props.mode === 'live' && props.$connected ? pulse : 'none'} 2s infinite;
`;

const StatusIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 500;
`;

const BotStatusDot = styled.div`
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: ${props => props.$active ? '#27ae60' : '#e74c3c'};
  animation: ${props => props.$active ? pulse : 'none'} 2s infinite;
`;

const SettingsButton = styled.button`
  background: ${props => props.disabled ? '#bdc3c7' : '#95a5a6'};
  color: white;
  border: none;
  padding: 10px;
  border-radius: 50%;
  cursor: ${props => props.disabled ? 'not-allowed' : 'pointer'};
  font-size: 1rem;
  transition: all 0.3s ease;
  opacity: ${props => props.disabled ? 0.6 : 1};

  &:hover {
    background: ${props => props.disabled ? '#bdc3c7' : '#7f8c8d'};
    transform: ${props => props.disabled ? 'none' : 'rotate(90deg)'};
  }
`;

const TabNavigation = styled.div`
  display: flex;
  background: white;
  border-radius: 10px;
  padding: 5px;
  margin-bottom: 20px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
`;

const TabButton = styled.button`
  flex: 1;
  background: ${props => props.$active ? '#3498db' : 'transparent'};
  border: none;
  padding: 12px 20px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: ${props => props.$active ? 'white' : '#7f8c8d'};
  box-shadow: ${props => props.$active ? '0 2px 8px rgba(52, 152, 219, 0.3)' : 'none'};

  &:hover:not(.active) {
    background: #ecf0f1;
    color: #2c3e50;
  }

  i {
    font-size: 1rem;
  }
`;

const ToastNotification = styled.div`
  position: fixed;
  top: 20px;
  right: 20px;
  background: #e74c3c;
  color: white;
  padding: 15px 20px;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  z-index: 1000;
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 500;
  animation: slideIn 0.3s ease-out;

  @keyframes slideIn {
    from {
      transform: translateX(100%);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }

  i {
    font-size: 1.2rem;
  }
`;

const Header = ({ botData, activeTab, onTabChange, onOpenSettings, liveStatus }) => {
  const [showToast, setShowToast] = useState(false);

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: 'fas fa-chart-line' },
    { id: 'portfolio', label: 'Portfolio', icon: 'fas fa-briefcase' },
    { id: 'chat', label: 'Chat Assistant', icon: 'fas fa-robot' }
  ];

  const handleSettingsClick = () => {
    if (botData.isRunning) {
      setShowToast(true);
      setTimeout(() => setShowToast(false), 3000); // Hide after 3 seconds
      return;
    }
    onOpenSettings();
  };

  return (
    <>
      {showToast && (
        <ToastNotification>
          <i className="fas fa-exclamation-triangle"></i>
          Cannot modify settings while bot is running! Stop the bot first.
        </ToastNotification>
      )}

      <HeaderContainer>
        <Title>💵BlackHole Trading Bot</Title>

        <HeaderControls>
          {/* Trading Mode Indicator */}
          {liveStatus && (
            <ModeIndicator mode={liveStatus.mode}>
              <StatusDot
                mode={liveStatus.mode}
                $connected={liveStatus.dhan_connected || liveStatus.mode === 'paper'}
              />
              <span>
                {liveStatus.mode === 'live' ? 'Live Trading' : 'Paper Trading'}
                {liveStatus.mode === 'live' && liveStatus.market_status && (
                  <small style={{ marginLeft: '4px', opacity: 0.8 }}>
                    ({liveStatus.market_status})
                  </small>
                )}
              </span>
            </ModeIndicator>
          )}

          <StatusIndicator>
            <BotStatusDot $active={botData.isRunning} />
            <span>{botData.isRunning ? 'Active' : 'Inactive'}</span>
          </StatusIndicator>

          <SettingsButton
            onClick={handleSettingsClick}
            disabled={botData.isRunning}
            title={botData.isRunning ? 'Settings disabled while bot is running' : 'Open Settings'}
          >
            <i className="fas fa-cog"></i>
          </SettingsButton>
        </HeaderControls>
      </HeaderContainer>

      <TabNavigation>
        {tabs.map(tab => (
          <TabButton
            key={tab.id}
            $active={activeTab === tab.id}
            onClick={() => onTabChange(tab.id)}
          >
            <i className={tab.icon}></i>
            {tab.label}
          </TabButton>
        ))}
      </TabNavigation>
    </>
  );
};

Header.propTypes = {
  botData: PropTypes.shape({
    isRunning: PropTypes.bool.isRequired
  }).isRequired,
  activeTab: PropTypes.string.isRequired,
  onTabChange: PropTypes.func.isRequired,
  onOpenSettings: PropTypes.func.isRequired,
  liveStatus: PropTypes.shape({
    mode: PropTypes.string,
    dhan_connected: PropTypes.bool,
    market_status: PropTypes.string
  })
};

export default Header;
