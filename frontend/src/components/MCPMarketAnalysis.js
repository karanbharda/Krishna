import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import apiService from '../services/apiService';

const AnalysisContainer = styled.div`
  background: white;
  border-radius: 15px;
  padding: 25px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  margin-bottom: 20px;
`;

const Header = styled.div`
  display: flex;
  justify-content: between;
  align-items: center;
  margin-bottom: 25px;
  flex-wrap: wrap;
  gap: 15px;
`;

const Title = styled.h2`
  color: #2c3e50;
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
`;

const AnalysisForm = styled.div`
  display: flex;
  gap: 15px;
  align-items: flex-end;
  flex-wrap: wrap;
`;

const InputGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 5px;
`;

const Label = styled.label`
  font-size: 0.9rem;
  font-weight: 500;
  color: #555;
`;

const Input = styled.input`
  padding: 10px 15px;
  border: 2px solid #e1e8ed;
  border-radius: 8px;
  font-size: 0.9rem;
  transition: border-color 0.3s ease;
  
  &:focus {
    outline: none;
    border-color: #667eea;
  }
`;

const Select = styled.select`
  padding: 10px 15px;
  border: 2px solid #e1e8ed;
  border-radius: 8px;
  font-size: 0.9rem;
  background: white;
  cursor: pointer;
  transition: border-color 0.3s ease;
  
  &:focus {
    outline: none;
    border-color: #667eea;
  }
`;

const AnalyzeButton = styled.button`
  padding: 10px 25px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const ResultsContainer = styled.div`
  margin-top: 25px;
`;

const RecommendationCard = styled.div`
  background: ${props => 
    props.recommendation === 'BUY' ? 'linear-gradient(135deg, #4CAF50 0%, #45a049 100%)' :
    props.recommendation === 'SELL' ? 'linear-gradient(135deg, #f44336 0%, #d32f2f 100%)' :
    'linear-gradient(135deg, #FF9800 0%, #F57C00 100%)'
  };
  color: white;
  padding: 20px;
  border-radius: 12px;
  margin-bottom: 20px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
`;

const RecommendationHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
`;

const RecommendationTitle = styled.h3`
  margin: 0;
  font-size: 1.3rem;
  font-weight: 600;
`;

const ConfidenceScore = styled.div`
  background: rgba(255, 255, 255, 0.2);
  padding: 8px 15px;
  border-radius: 20px;
  font-weight: 600;
  font-size: 0.9rem;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-top: 20px;
`;

const MetricCard = styled.div`
  background: #f8f9fa;
  padding: 15px;
  border-radius: 10px;
  border-left: 4px solid #667eea;
`;

const MetricLabel = styled.div`
  font-size: 0.8rem;
  color: #666;
  margin-bottom: 5px;
  font-weight: 500;
`;

const MetricValue = styled.div`
  font-size: 1.1rem;
  font-weight: 600;
  color: #2c3e50;
`;

const ReasoningSection = styled.div`
  background: #f8f9fa;
  padding: 20px;
  border-radius: 10px;
  margin-top: 20px;
`;

const ReasoningTitle = styled.h4`
  margin: 0 0 15px 0;
  color: #2c3e50;
  font-size: 1.1rem;
`;

const ReasoningText = styled.p`
  margin: 0;
  line-height: 1.6;
  color: #555;
`;

const LoadingSpinner = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 40px;
  
  &::after {
    content: '';
    width: 40px;
    height: 40px;
    border: 4px solid #f3f3f3;
    border-top: 4px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const ErrorMessage = styled.div`
  background: #ffebee;
  color: #c62828;
  padding: 15px;
  border-radius: 8px;
  border-left: 4px solid #f44336;
  margin-top: 15px;
`;

const MCPMarketAnalysis = () => {
  const [symbol, setSymbol] = useState('NSE:RELIANCE-EQ');
  const [timeframe, setTimeframe] = useState('1D');
  const [analysisType, setAnalysisType] = useState('comprehensive');
  const [isLoading, setIsLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState(null);
  const [mcpStatus, setMcpStatus] = useState('unknown');

  useEffect(() => {
    checkMcpStatus();
  }, []);

  const checkMcpStatus = async () => {
    try {
      const status = await apiService.getMcpStatus();
      setMcpStatus(status.mcp_available ? 'available' : 'unavailable');
    } catch (error) {
      setMcpStatus('unavailable');
    }
  };

  const handleAnalyze = async () => {
    if (!symbol.trim()) return;

    setIsLoading(true);
    setError(null);
    setAnalysisResult(null);

    try {
      const analysisRequest = {
        symbol: symbol.trim(),
        timeframe,
        analysis_type: analysisType
      };

      const result = await apiService.mcpAnalyzeMarket(analysisRequest);
      setAnalysisResult(result);
    } catch (error) {
      console.error('Analysis error:', error);
      setError(error.response?.data?.detail || 'Failed to analyze market. Please check if MCP server is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const formatMetricValue = (value, type = 'number') => {
    if (value === null || value === undefined) return 'N/A';
    
    switch (type) {
      case 'percentage':
        return `${(value * 100).toFixed(2)}%`;
      case 'currency':
        return `₹${value.toFixed(2)}`;
      case 'score':
        return value.toFixed(3);
      default:
        return typeof value === 'number' ? value.toFixed(2) : value;
    }
  };

  const getRecommendationIcon = (recommendation) => {
    switch (recommendation) {
      case 'BUY': return '📈';
      case 'SELL': return '📉';
      case 'HOLD': return '⏸️';
      default: return '⏳';
    }
  };

  return (
    <AnalysisContainer>
      <Header>
        <Title>🧠 AI Market Analysis</Title>
        <div style={{ fontSize: '0.9rem', color: mcpStatus === 'available' ? '#4CAF50' : '#f44336' }}>
          MCP Status: {mcpStatus === 'available' ? '✅ Connected' : '❌ Disconnected'}
        </div>
      </Header>

      <AnalysisForm>
        <InputGroup>
          <Label>Stock Symbol</Label>
          <Input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            placeholder="e.g., NSE:RELIANCE-EQ"
            disabled={isLoading}
          />
        </InputGroup>

        <InputGroup>
          <Label>Timeframe</Label>
          <Select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            disabled={isLoading}
          >
            <option value="1D">1 Day</option>
            <option value="1W">1 Week</option>
            <option value="1M">1 Month</option>
          </Select>
        </InputGroup>

        <InputGroup>
          <Label>Analysis Type</Label>
          <Select
            value={analysisType}
            onChange={(e) => setAnalysisType(e.target.value)}
            disabled={isLoading}
          >
            <option value="quick">Quick Analysis</option>
            <option value="comprehensive">Comprehensive</option>
            <option value="deep">Deep Analysis</option>
          </Select>
        </InputGroup>

        <AnalyzeButton 
          onClick={handleAnalyze} 
          disabled={isLoading || !symbol.trim() || mcpStatus !== 'available'}
        >
          {isLoading ? 'Analyzing...' : 'Analyze'}
        </AnalyzeButton>
      </AnalysisForm>

      {error && <ErrorMessage>{error}</ErrorMessage>}

      {isLoading && <LoadingSpinner />}

      {analysisResult && (
        <ResultsContainer>
          <RecommendationCard recommendation={analysisResult.recommendation}>
            <RecommendationHeader>
              <RecommendationTitle>
                {getRecommendationIcon(analysisResult.recommendation)} {analysisResult.recommendation}
              </RecommendationTitle>
              <ConfidenceScore>
                Confidence: {formatMetricValue(analysisResult.confidence, 'percentage')}
              </ConfidenceScore>
            </RecommendationHeader>
          </RecommendationCard>

          <MetricsGrid>
            <MetricCard>
              <MetricLabel>Current Price</MetricLabel>
              <MetricValue>{formatMetricValue(analysisResult.current_price, 'currency')}</MetricValue>
            </MetricCard>

            <MetricCard>
              <MetricLabel>Target Price</MetricLabel>
              <MetricValue>{formatMetricValue(analysisResult.target_price, 'currency')}</MetricValue>
            </MetricCard>

            <MetricCard>
              <MetricLabel>Stop Loss</MetricLabel>
              <MetricValue>{formatMetricValue(analysisResult.stop_loss, 'currency')}</MetricValue>
            </MetricCard>

            <MetricCard>
              <MetricLabel>Risk Score</MetricLabel>
              <MetricValue>{formatMetricValue(analysisResult.risk_score, 'score')}</MetricValue>
            </MetricCard>

            <MetricCard>
              <MetricLabel>Position Size</MetricLabel>
              <MetricValue>{formatMetricValue(analysisResult.position_size, 'percentage')}</MetricValue>
            </MetricCard>

            <MetricCard>
              <MetricLabel>Expected Return</MetricLabel>
              <MetricValue>{formatMetricValue(analysisResult.expected_return, 'percentage')}</MetricValue>
            </MetricCard>
          </MetricsGrid>

          {analysisResult.reasoning && (
            <ReasoningSection>
              <ReasoningTitle>🧠 AI Reasoning</ReasoningTitle>
              <ReasoningText>{analysisResult.reasoning}</ReasoningText>
            </ReasoningSection>
          )}

          {analysisResult.metadata && (
            <ReasoningSection>
              <ReasoningTitle>📊 Technical Analysis</ReasoningTitle>
              <MetricsGrid>
                {analysisResult.metadata.technical_signals && Object.entries(analysisResult.metadata.technical_signals).map(([category, signals]) => (
                  <MetricCard key={category}>
                    <MetricLabel>{category.charAt(0).toUpperCase() + category.slice(1)} Signals</MetricLabel>
                    <MetricValue>
                      {typeof signals === 'object' && signals.composite_score 
                        ? formatMetricValue(signals.composite_score, 'score')
                        : 'Processing...'}
                    </MetricValue>
                  </MetricCard>
                ))}
              </MetricsGrid>
            </ReasoningSection>
          )}
        </ResultsContainer>
      )}
    </AnalysisContainer>
  );
};

export default MCPMarketAnalysis;
