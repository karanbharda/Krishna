import React from 'react';
import styled from 'styled-components';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, ArcElement } from 'chart.js';
import { Line, Doughnut } from 'react-chartjs-2';
import { formatCurrency } from '../services/apiService';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, ArcElement);

const DashboardContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 30px;
`;

const MetricsRow = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
`;

const MetricCard = styled.div`
  background: white;
  color: #333;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  padding: 20px;
  border-radius: 12px;
  text-align: center;
`;

const MetricLabel = styled.div`
  font-size: 0.9rem;
  color: #7f8c8d;
  margin-bottom: 8px;
`;

const MetricValue = styled.div`
  font-size: 1.8rem;
  font-weight: bold;
  color: #2c3e50;
`;

const ChartsSection = styled.div`
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 20px;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const ChartContainer = styled.div`
  background: #f8f9fa;
  padding: 20px;
  border-radius: 10px;
  border: 1px solid #e9ecef;

  h3 {
    margin-bottom: 15px;
    color: #2c3e50;
    text-align: center;
  }
`;

const ActivitySection = styled.div`
  background: #f8f9fa;
  padding: 20px;
  border-radius: 10px;
  border: 1px solid #e9ecef;

  h3 {
    margin-bottom: 15px;
    color: #2c3e50;
  }
`;

const ActivityList = styled.div`
  max-height: 300px;
  overflow-y: auto;
`;

const ActivityItem = styled.div`
  background: white;
  padding: 15px;
  margin-bottom: 10px;
  border-radius: 8px;
  border-left: 4px solid ${props => props.type === 'buy' ? '#27ae60' : '#e74c3c'};
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const ActivityDetails = styled.div`
  flex: 1;
`;

const ActivityTitle = styled.div`
  font-weight: bold;
  margin-bottom: 5px;
`;

const ActivityTime = styled.div`
  font-size: 0.9rem;
  color: #666;
`;

const ActivityValues = styled.div`
  text-align: right;
  
  div {
    margin-bottom: 2px;
    
    strong {
      font-weight: 600;
    }
  }
`;

const NoActivity = styled.div`
  text-align: center;
  color: #7f8c8d;
  font-style: italic;
  padding: 20px;
`;

const Dashboard = ({ botData }) => {
  const calculateMetrics = () => {
    const totalValue = botData.portfolio.totalValue;
    const cash = botData.portfolio.cash;
    const startingBalance = botData.portfolio.startingBalance;
    const totalReturn = totalValue - startingBalance;
    
    // Calculate unrealized P&L (simplified)
    let unrealizedPnL = 0;
    Object.values(botData.portfolio.holdings).forEach(holding => {
      unrealizedPnL += (holding.currentPrice || holding.avgPrice) * holding.qty - holding.avgPrice * holding.qty;
    });
    
    const tradesToday = botData.portfolio.tradeLog.filter(trade => 
      trade.timestamp.startsWith(new Date().toISOString().split('T')[0])
    ).length;
    
    return {
      totalValue,
      unrealizedPnL,
      activePositions: Object.keys(botData.portfolio.holdings).length,
      tradesToday
    };
  };

  const generatePortfolioChartData = () => {
    // Generate sample data for the last 30 days
    const labels = [];
    const data = [];
    const baseValue = botData.portfolio.startingBalance;
    
    for (let i = 29; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      labels.push(date.toLocaleDateString());
      
      // Simulate portfolio value changes
      const variation = (Math.random() - 0.5) * 0.02; // ±1% daily variation
      const value = baseValue * (1 + variation * (30 - i) / 30);
      data.push(value);
    }
    
    return {
      labels,
      datasets: [{
        label: 'Portfolio Value',
        data,
        borderColor: '#3498db',
        backgroundColor: 'rgba(52, 152, 219, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4
      }]
    };
  };

  const generateAllocationChartData = () => {
    const holdings = botData.portfolio.holdings;
    const labels = Object.keys(holdings).length > 0 ? Object.keys(holdings) : ['Cash'];
    const data = Object.keys(holdings).length > 0 
      ? Object.values(holdings).map(h => h.qty * h.avgPrice)
      : [botData.portfolio.cash];
    
    return {
      labels,
      datasets: [{
        data,
        backgroundColor: [
          '#3498db', '#e74c3c', '#f39c12', '#27ae60', '#9b59b6',
          '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#2ecc71'
        ]
      }]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: false,
        ticks: {
          callback: function(value) {
            return '₹' + (value / 100000).toFixed(1) + 'L';
          }
        }
      }
    },
    plugins: {
      legend: {
        display: false
      }
    }
  };

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom'
      }
    }
  };

  const metrics = calculateMetrics();
  const portfolioChartData = generatePortfolioChartData();
  const allocationChartData = generateAllocationChartData();
  const recentTrades = botData.portfolio.tradeLog.slice(-10).reverse();

  return (
    <DashboardContainer>
      {/* Performance Metrics */}
      <MetricsRow>
        <MetricCard>
          <MetricLabel>Portfolio Value</MetricLabel>
          <MetricValue>{formatCurrency(metrics.totalValue)}</MetricValue>
        </MetricCard>
        
        <MetricCard>
          <MetricLabel>Unrealized P&L</MetricLabel>
          <MetricValue>{formatCurrency(metrics.unrealizedPnL)}</MetricValue>
        </MetricCard>
        
        <MetricCard>
          <MetricLabel>Active Positions</MetricLabel>
          <MetricValue>{metrics.activePositions}</MetricValue>
        </MetricCard>
        
        <MetricCard>
          <MetricLabel>Trades Today</MetricLabel>
          <MetricValue>{metrics.tradesToday}</MetricValue>
        </MetricCard>
      </MetricsRow>

      {/* Charts Section */}
      <ChartsSection>
        <ChartContainer>
          <h3>Portfolio Performance</h3>
          <div style={{ height: '300px' }}>
            <Line data={portfolioChartData} options={chartOptions} />
          </div>
        </ChartContainer>
        
        <ChartContainer>
          <h3>Asset Allocation</h3>
          <div style={{ height: '300px' }}>
            <Doughnut data={allocationChartData} options={doughnutOptions} />
          </div>
        </ChartContainer>
      </ChartsSection>

      {/* Recent Activity */}
      <ActivitySection>
        <h3>Recent Trading Activity</h3>
        <ActivityList>
          {recentTrades.length > 0 ? (
            recentTrades.map((trade, index) => (
              <ActivityItem key={index} type={trade.action}>
                <ActivityDetails>
                  <ActivityTitle>
                    {trade.action.toUpperCase()} {trade.asset}
                  </ActivityTitle>
                  <ActivityTime>
                    {new Date(trade.timestamp).toLocaleString()}
                  </ActivityTime>
                </ActivityDetails>
                <ActivityValues>
                  <div><strong>Qty:</strong> {trade.qty}</div>
                  <div><strong>Price:</strong> {formatCurrency(trade.price)}</div>
                  <div><strong>Value:</strong> {formatCurrency(trade.qty * trade.price)}</div>
                </ActivityValues>
              </ActivityItem>
            ))
          ) : (
            <NoActivity>No recent trades</NoActivity>
          )}
        </ActivityList>
      </ActivitySection>
    </DashboardContainer>
  );
};

export default Dashboard;
