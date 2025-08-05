import React, { useState } from 'react';
import styled from 'styled-components';
import PropTypes from 'prop-types';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  Filler
} from 'chart.js';
import { Line, Doughnut } from 'react-chartjs-2';
import { formatCurrency } from '../services/apiService';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, ArcElement, Filler);

const DashboardContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
  min-height: 100%;
  overflow: visible;
  padding-bottom: 20px;
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

const ChartHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;

  h3 {
    margin: 0;
    color: #2c3e50;
  }
`;

const TimePeriodButtons = styled.div`
  display: flex;
  gap: 5px;
`;

const TimePeriodButton = styled.button`
  padding: 6px 12px;
  border: 1px solid #ddd;
  background: ${props => props.$active ? '#f39c12' : 'white'};
  color: ${props => props.$active ? 'white' : '#666'};
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: ${props => props.$active ? '#e67e22' : '#f8f9fa'};
    border-color: #f39c12;
  }

  &:focus {
    outline: none;
    box-shadow: 0 0 0 2px rgba(243, 156, 18, 0.2);
  }
`;

const PortfolioValueDisplay = styled.div`
  margin-bottom: 20px;
  padding: 0 5px;
`;

const PortfolioValue = styled.div`
  font-size: 28px;
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 5px;
`;

const PortfolioChange = styled.span`
  font-size: 16px;
  font-weight: 500;
  color: ${props => props.$positive ? '#27ae60' : '#e74c3c'};
`;

const PortfolioTimestamp = styled.div`
  font-size: 12px;
  color: #95a5a6;
  margin-top: 5px;
`;



const ActivitySection = styled.div`
  background: #f8f9fa;
  padding: 20px;
  border-radius: 10px;
  border: 1px solid #e9ecef;
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 450px;
  max-height: 550px;
  height: 450px;

  h3 {
    margin-bottom: 15px;
    color: #2c3e50;
    flex-shrink: 0;
  }
`;

const ActivityList = styled.div`
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  max-height: 350px;
  min-height: 200px;
  padding-right: 8px;
  margin-right: -8px;

  /* Custom scrollbar - Always visible */
  &::-webkit-scrollbar {
    width: 8px;
    background: transparent;
  }

  &::-webkit-scrollbar-track {
    background: #f8f9fa;
    border-radius: 4px;
    border: 1px solid #e9ecef;
  }

  &::-webkit-scrollbar-thumb {
    background: #6c757d;
    border-radius: 4px;
    border: 1px solid #dee2e6;
  }

  &::-webkit-scrollbar-thumb:hover {
    background: #495057;
  }

  &::-webkit-scrollbar-thumb:active {
    background: #343a40;
  }

  /* Firefox scrollbar */
  scrollbar-width: thin;
  scrollbar-color: #6c757d #f8f9fa;
`;

const ActivityItem = styled.div`
  background: white;
  padding: 16px;
  margin-bottom: 10px;
  border-radius: 8px;
  border-left: 4px solid ${props => props.type === 'buy' ? '#27ae60' : '#e74c3c'};
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  min-height: 70px;
  flex-shrink: 0;

  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  &:last-child {
    margin-bottom: 10px;
  }
`;

const ActivityDetails = styled.div`
  flex: 1;
`;

const ActivityTitle = styled.div`
  font-weight: bold;
  margin-bottom: 5px;
  color: ${props => props.type === 'buy' ? '#27ae60' : '#e74c3c'};
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

  div {
    margin-bottom: 3px;

    &:last-child {
      margin-bottom: 0;
      font-weight: bold;
      color: #2c3e50;
    }

    strong {
      font-weight: 600;
    }
  }
`;

const NoActivity = styled.div`
  text-align: center;
  color: #7f8c8d;
  font-style: italic;
  padding: 40px 20px;
  background: white;
  border-radius: 8px;
  border: 2px dashed #e9ecef;
`;

const Dashboard = ({ botData }) => {
  const [timePeriod, setTimePeriod] = useState('1M');

  const getPeriodDescription = () => {
    switch (timePeriod) {
      case '1D': return 'Last 24 Hours';
      case '1M': return 'Last 30 Days';
      case '1Y': return 'Last 12 Months';
      case 'All': return 'Last 2 Years';
      default: return 'Last 30 Days';
    }
  };

  const calculateMetrics = () => {
    const totalValue = botData.portfolio.totalValue || 0;
    const startingBalance = botData.portfolio.startingBalance || 10000;

    // Use unrealized P&L from backend if available, otherwise calculate
    let unrealizedPnL = 0;
    if (botData.portfolio.unrealizedPnL !== undefined) {
      unrealizedPnL = botData.portfolio.unrealizedPnL;
    } else {
      // Fallback calculation with proper null checks
      Object.values(botData.portfolio.holdings || {}).forEach(holding => {
        const currentPrice = holding.currentPrice || holding.avgPrice || 0;
        const avgPrice = holding.avgPrice || 0;
        const qty = holding.qty || 0;
        unrealizedPnL += (currentPrice - avgPrice) * qty;
      });
    }

    const tradesToday = (botData.portfolio.tradeLog || []).filter(trade =>
      trade.timestamp && trade.timestamp.startsWith(new Date().toISOString().split('T')[0])
    ).length;

    return {
      totalValue,
      unrealizedPnL,
      activePositions: Object.keys(botData.portfolio.holdings || {}).length,
      tradesToday
    };
  };

  const generatePortfolioChartData = () => {
    const startingBalance = botData.portfolio.startingBalance || 10000;
    const currentValue = botData.portfolio.totalValue || startingBalance;

    // Generate time series data based on selected period
    const generateTimeSeriesData = () => {
      const labels = [];
      const values = [];
      let daysToShow = 30;

      // Determine period based on timePeriod state
      switch (timePeriod) {
        case '1D':
          daysToShow = 1;
          break;
        case '1M':
          daysToShow = 30;
          break;
        case '1Y':
          daysToShow = 365;
          break;
        case 'All':
          daysToShow = 730; // 2 years
          break;
        default:
          daysToShow = 30;
      }

      if (timePeriod === '1D') {
        // Generate hourly data for 1 day
        for (let i = 23; i >= 0; i--) {
          const date = new Date();
          date.setHours(date.getHours() - i);
          labels.push(date.toLocaleTimeString('en-IN', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: false
          }));

          // Generate realistic hourly fluctuations
          const hourProgress = (23 - i) / 23;
          const trend = (currentValue - startingBalance) * hourProgress * 0.1; // Smaller daily trend
          const randomFluctuation = (Math.random() - 0.5) * (startingBalance * 0.005); // ±0.5% hourly fluctuation
          const value = Math.max(startingBalance + trend + randomFluctuation, startingBalance * 0.95);

          values.push(Math.round(value));
        }
      } else if (timePeriod === '1Y' || timePeriod === 'All') {
        // Generate monthly data for longer periods
        const monthsToShow = timePeriod === '1Y' ? 12 : 24;
        for (let i = monthsToShow - 1; i >= 0; i--) {
          const date = new Date();
          date.setMonth(date.getMonth() - i);
          labels.push(date.toLocaleDateString('en-IN', {
            month: 'short',
            year: '2-digit'
          }));

          // Generate realistic monthly progression
          const monthProgress = (monthsToShow - 1 - i) / (monthsToShow - 1);
          const trend = (currentValue - startingBalance) * monthProgress;
          const randomFluctuation = (Math.random() - 0.5) * (startingBalance * 0.08); // ±8% monthly fluctuation
          const value = Math.max(startingBalance + trend + randomFluctuation, startingBalance * 0.7);

          values.push(Math.round(value));
        }
      } else {
        // Generate daily data for 1M
        for (let i = daysToShow - 1; i >= 0; i--) {
          const date = new Date();
          date.setDate(date.getDate() - i);
          labels.push(date.toLocaleDateString('en-IN', {
            month: 'short',
            day: 'numeric'
          }));

          // Generate realistic daily fluctuations
          const dayProgress = (daysToShow - 1 - i) / (daysToShow - 1);
          const trend = (currentValue - startingBalance) * dayProgress;
          const randomFluctuation = (Math.random() - 0.5) * (startingBalance * 0.03); // ±3% daily fluctuation
          const value = Math.max(startingBalance + trend + randomFluctuation, startingBalance * 0.8);

          values.push(Math.round(value));
        }
      }

      // Ensure the last value matches current portfolio value
      values[values.length - 1] = currentValue;

      return { labels, values };
    };

    const { labels, values } = generateTimeSeriesData();

    // Return area chart configuration with the specified blue color
    return {
      labels,
      datasets: [{
        label: 'Portfolio Value (₹)',
        data: values,
        borderColor: '#4A90E2',
        backgroundColor: 'rgba(74, 144, 226, 0.2)',
        fill: true,
        borderWidth: 2,
        tension: 0.3,
        pointRadius: 0,
        pointHoverRadius: 0,
        pointBackgroundColor: 'transparent',
        pointBorderColor: 'transparent'
      }]
    };
  };

  const generateAllocationChartData = () => {
    const holdings = botData.portfolio.holdings || {};
    const cash = botData.portfolio.cash || 0;

    const labels = [];
    const data = [];
    const colors = ['#3498db', '#e74c3c', '#f39c12', '#27ae60', '#9b59b6', '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#2ecc71'];

    // Add holdings with current market values
    Object.entries(holdings).forEach(([ticker, holding]) => {
      const currentPrice = holding.currentPrice || holding.avgPrice || 0;
      const currentValue = holding.qty * currentPrice;
      if (currentValue > 0) {
        labels.push(ticker);
        data.push(currentValue);
      }
    });

    // Add cash if significant
    if (cash > 0) {
      labels.push('Cash');
      data.push(cash);
    }

    // If no data, show placeholder
    if (labels.length === 0) {
      labels.push('No Holdings');
      data.push(1);
    }

    return {
      labels,
      datasets: [{
        data,
        backgroundColor: colors.slice(0, labels.length)
      }]
    };
  };



  const getChartOptions = () => {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        intersect: false,
        mode: 'index'
      },
      scales: {
        x: {
          display: true,
          ticks: {
            maxTicksLimit: timePeriod === '1D' ? 8 : timePeriod === '1M' ? 6 : 5,
            font: {
              size: 10
            }
          },
          grid: {
            display: false
          },
          title: {
            display: false
          }
        },
        y: {
          display: true,
          beginAtZero: false,
          ticks: {
            callback: function (value) {
              if (value >= 100000) {
                return '₹' + (value / 100000).toFixed(1) + 'L';
              } else if (value >= 1000) {
                return '₹' + (value / 1000).toFixed(1) + 'K';
              } else {
                return '₹' + value.toFixed(0);
              }
            },
            font: {
              size: 10
            }
          },
          grid: {
            color: 'rgba(0, 0, 0, 0.05)',
            drawBorder: false
          },
          title: {
            display: false
          }
        }
      },
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          titleColor: 'white',
          bodyColor: 'white',
          borderColor: '#4A90E2',
          borderWidth: 1,
          cornerRadius: 6,
          displayColors: false,
          callbacks: {
            title: function (context) {
              return context[0].label;
            },
            label: function (context) {
              return `₹${context.parsed.y.toLocaleString('en-IN')}`;
            }
          }
        },
        filler: {
          propagate: true
        }
      },
      elements: {
        line: {
          tension: 0.3
        }
      }
    };
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

  // Get recent trades and remove any duplicates
  const allTrades = (botData.portfolio.tradeLog || []);

  // Remove duplicates based on asset, timestamp, qty, and price
  const uniqueTrades = allTrades.filter((trade, index, self) =>
    index === self.findIndex(t =>
      t.asset === trade.asset &&
      t.timestamp === trade.timestamp &&
      t.qty === trade.qty &&
      t.price === trade.price &&
      t.action === trade.action
    )
  );

  let recentTrades = uniqueTrades.slice(-20).reverse();

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
          <ChartHeader>
            <h3>Portfolio Performance</h3>
            <TimePeriodButtons>
              {['1D', '1M', '1Y', 'All'].map(period => (
                <TimePeriodButton
                  key={period}
                  $active={timePeriod === period}
                  onClick={() => setTimePeriod(period)}
                >
                  {period}
                </TimePeriodButton>
              ))}
            </TimePeriodButtons>
          </ChartHeader>
          <PortfolioValueDisplay>
            <PortfolioValue>
              ₹ {metrics.totalValue.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              <PortfolioChange $positive={metrics.unrealizedPnL >= 0}>
                {metrics.unrealizedPnL >= 0 ? ' +' : ' '}
                {((metrics.unrealizedPnL / (metrics.totalValue - metrics.unrealizedPnL)) * 100).toFixed(2)}%
              </PortfolioChange>
            </PortfolioValue>
            <PortfolioTimestamp>
              {getPeriodDescription()} • {new Date().toLocaleDateString('en-IN', {
                month: 'short',
                day: '2-digit',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                hour12: true
              })}
            </PortfolioTimestamp>
          </PortfolioValueDisplay>
          <div style={{ height: '280px' }}>
            <Line data={portfolioChartData} options={getChartOptions()} />
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
            recentTrades.map((trade, index) => {
              // Handle different possible data structures
              const action = trade.action || 'unknown';
              const asset = trade.asset || trade.ticker || 'Unknown';
              const qty = trade.qty || trade.quantity || 0;
              const price = trade.price || 0;
              const timestamp = trade.timestamp || new Date().toISOString();

              // Format the asset name (remove .NS suffix for display)
              const displayAsset = asset.replace('.NS', '');

              // Format the timestamp
              const tradeDate = new Date(timestamp);
              const formattedTime = tradeDate.toLocaleDateString('en-IN') + ' ' +
                tradeDate.toLocaleTimeString('en-IN', {
                  hour: '2-digit',
                  minute: '2-digit'
                });

              return (
                <ActivityItem key={`trade-${index}-${trade.asset}-${trade.timestamp}-${trade.qty}-${trade.price}`} type={action}>
                  <ActivityDetails>
                    <ActivityTitle type={action}>
                      {action.toUpperCase()} {displayAsset}
                    </ActivityTitle>
                    <ActivityTime>
                      {formattedTime}
                    </ActivityTime>
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
            <NoActivity>
              <div>📈</div>
              <div style={{ marginTop: '10px' }}>No recent trades</div>
              <div style={{ fontSize: '0.8rem', marginTop: '5px' }}>
                Start trading to see your activity here
              </div>
            </NoActivity>
          )}
        </ActivityList>
      </ActivitySection>
    </DashboardContainer>
  );
};

Dashboard.propTypes = {
  botData: PropTypes.shape({
    portfolio: PropTypes.object.isRequired
  }).isRequired
};

export default Dashboard;
