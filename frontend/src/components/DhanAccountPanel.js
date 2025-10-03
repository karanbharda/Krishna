import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { formatCurrency, formatPercentage } from '../services/apiService';

const DhanAccountPanel = styled.div`
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  padding: 20px;
  margin-bottom: 20px;
`;

const PanelHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 10px;
  border-bottom: 1px solid #eee;

  h3 {
    margin: 0;
    color: #2c3e50;
    display: flex;
    align-items: center;
    gap: 8px;
  }
`;

const StatusBadge = styled.span`
  background: ${props => props.$isLive ? '#2ecc71' : '#95a5a6'};
  color: white;
  font-size: 0.7rem;
  padding: 3px 8px;
  border-radius: 10px;
  font-weight: 600;
  text-transform: uppercase;
`;

const RefreshButton = styled.button`
  background: #3498db;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 6px 12px;
  font-size: 0.8rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: background 0.2s;

  &:hover {
    background: #2980b9;
  }
`;

const AccountSummary = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 20px;
`;

const SummaryCard = styled.div`
  background: #f8f9fa;
  border-radius: 8px;
  padding: 15px;
  text-align: center;
`;

const SummaryLabel = styled.div`
  font-size: 0.8rem;
  color: #7f8c8d;
  margin-bottom: 5px;
`;

const SummaryValue = styled.div`
  font-size: 1.3rem;
  font-weight: 600;
  color: #2c3e50;
`;

const PnLValue = styled(SummaryValue)`
  color: ${props => props.$isPositive ? (props.$isZero ? '#000' : '#27ae60') : '#e74c3c'};
`;

const SectionTitle = styled.h4`
  margin: 20px 0 10px 0;
  color: #34495e;
  font-size: 1rem;
`;

const PositionsTable = styled.div`
  width: 100%;
  overflow-x: auto;
  margin-bottom: 20px;
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;

  th, td {
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid #eee;
  }

  th {
    background: #f8f9fa;
    font-weight: 600;
    color: #7f8c8d;
    text-transform: uppercase;
    font-size: 0.7rem;
    letter-spacing: 0.5px;
  }

  tr:hover {
    background: #f8f9fa;
  }
`;

const ErrorMessage = styled.div`
  color: #e74c3c;
  background: #fde8e8;
  padding: 10px 15px;
  border-radius: 4px;
  margin: 10px 0;
  font-size: 0.85rem;
`;

const DhanAccount = ({ dhanData, onRefresh, isLive }) => {
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    setLastUpdated(new Date());
  }, [dhanData]);

  const handleRefresh = async () => {
    try {
      setIsRefreshing(true);
      await onRefresh();
    } finally {
      setIsRefreshing(false);
    }
  };

  if (!dhanData) {
    return (
      <DhanAccountPanel>
        <PanelHeader>
          <h3>Dhan Account <StatusBadge $isLive={isLive}>{isLive ? 'Live' : 'Paper'}</StatusBadge></h3>
          <RefreshButton onClick={handleRefresh} disabled={isRefreshing}>
            {isRefreshing ? 'Refreshing...' : 'Refresh'}
          </RefreshButton>
        </PanelHeader>
        <p>No Dhan account data available. Make sure you're connected to Dhan API.</p>
      </DhanAccountPanel>
    );
  }

  const { funds, positions = [], orders = [], trades = [], error } = dhanData;
  const totalPnl = positions.reduce((sum, pos) => sum + (parseFloat(pos.pnl) || 0), 0);
  const totalPnlPct = positions.length > 0 
    ? (totalPnl / positions.reduce((sum, pos) => sum + (parseFloat(pos.buy_value) || 0), 0)) * 100 
    : 0;

  return (
    <DhanAccountPanel>
      <PanelHeader>
        <h3>
          Dhan Account
          <StatusBadge $isLive={isLive}>{isLive ? 'Live' : 'Paper'}</StatusBadge>
        </h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '0.8rem', color: '#7f8c8d' }}>
            Last updated: {lastUpdated.toLocaleTimeString()}
          </span>
          <RefreshButton onClick={handleRefresh} disabled={isRefreshing}>
            <i className={`fas fa-sync ${isRefreshing ? 'fa-spin' : ''}`}></i>
            {isRefreshing ? 'Refreshing...' : 'Refresh'}
          </RefreshButton>
        </div>
      </PanelHeader>

      {error && <ErrorMessage>Error: {error}</ErrorMessage>}

      {funds && (
        <>
          <AccountSummary>
            <SummaryCard>
              <SummaryLabel>Available Balance</SummaryLabel>
              <SummaryValue>{formatCurrency(funds.available_balance)}</SummaryValue>
            </SummaryCard>
            <SummaryCard>
              <SummaryLabel>Used Margin</SummaryLabel>
              <SummaryValue>{formatCurrency(funds.used_margin)}</SummaryValue>
            </SummaryCard>
            <SummaryCard>
              <SummaryLabel>Net Balance</SummaryLabel>
              <SummaryValue>{formatCurrency(funds.net_balance)}</SummaryValue>
            </SummaryCard>
            <SummaryCard>
              <SummaryLabel>Total P&L</SummaryLabel>
              <PnLValue 
                $isPositive={totalPnl >= 0} 
                $isZero={totalPnl === 0}
              >
                {formatCurrency(totalPnl)} ({formatPercentage(totalPnlPct)})
              </PnLValue>
            </SummaryCard>
          </AccountSummary>

          {positions.length > 0 && (
            <>
              <SectionTitle>Open Positions</SectionTitle>
              <PositionsTable>
                <Table>
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Qty</th>
                      <th>Avg. Price</th>
                      <th>LTP</th>
                      <th>P&L</th>
                      <th>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((pos, idx) => (
                      <tr key={`${pos.symbol}-${idx}`}>
                        <td>{pos.symbol}</td>
                        <td>{pos.quantity}</td>
                        <td>{formatCurrency(pos.avg_price)}</td>
                        <td>{formatCurrency(pos.ltp)}</td>
                        <td style={{ color: pos.pnl >= 0 ? '#27ae60' : '#e74c3c' }}>
                          {formatCurrency(pos.pnl)} ({formatPercentage(pos.pnl_pct)})
                        </td>
                        <td>{formatCurrency(pos.current_value)}</td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </PositionsTable>
            </>
          )}

          {orders.length > 0 && (
            <>
              <SectionTitle>Orders</SectionTitle>
              <PositionsTable>
                <Table>
                  <thead>
                    <tr>
                      <th>Order ID</th>
                      <th>Symbol</th>
                      <th>Type</th>
                      <th>Qty</th>
                      <th>Price</th>
                      <th>Status</th>
                      <th>Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {orders.slice(0, 5).map((order) => (
                      <tr key={order.order_id}>
                        <td>{order.order_id}</td>
                        <td>{order.trading_symbol}</td>
                        <td>{order.transaction_type}</td>
                        <td>{order.quantity}</td>
                        <td>{formatCurrency(order.price)}</td>
                        <td>
                          <span style={{
                            color: order.status === 'COMPLETE' ? '#27ae60' : 
                                  order.status === 'REJECTED' ? '#e74c3c' : '#f39c12',
                            fontWeight: 500
                          }}>
                            {order.status}
                          </span>
                        </td>
                        <td>{new Date(order.order_timestamp).toLocaleTimeString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </PositionsTable>
            </>
          )}
        </>
      )}
    </DhanAccountPanel>
  );
};

export default DhanAccount;
