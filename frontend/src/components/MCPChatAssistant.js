import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import { apiService } from '../services/apiService';

const ChatContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 15px;
  overflow: hidden;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
`;

const ChatHeader = styled.div`
  background: rgba(255, 255, 255, 0.1);
  padding: 20px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
  backdrop-filter: blur(10px);
`;

const HeaderTitle = styled.h3`
  color: white;
  margin: 0 0 5px 0;
  font-size: 1.2rem;
  font-weight: 600;
`;

const HeaderSubtitle = styled.p`
  color: rgba(255, 255, 255, 0.8);
  margin: 0;
  font-size: 0.9rem;
`;

const StatusIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
`;

const StatusDot = styled.div`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${props => props.status === 'connected' ? '#4CAF50' :
    props.status === 'connecting' ? '#FF9800' : '#F44336'};
  animation: ${props => props.status === 'connecting' ? 'pulse 1.5s infinite' : 'none'};
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
`;

const StatusText = styled.span`
  color: rgba(255, 255, 255, 0.9);
  font-size: 0.8rem;
`;

const MessagesContainer = styled.div`
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 15px;
`;

const Message = styled.div`
  display: flex;
  flex-direction: column;
  align-items: ${props => props.isUser ? 'flex-end' : 'flex-start'};
`;

const MessageBubble = styled.div`
  max-width: 80%;
  padding: 12px 16px;
  border-radius: 18px;
  background: ${props => props.isUser
    ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    : 'rgba(255, 255, 255, 0.95)'};
  color: ${props => props.isUser ? 'white' : '#333'};
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  word-wrap: break-word;
  line-height: 1.4;
`;

const MessageMeta = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 5px;
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.7);
`;

const ConfidenceBar = styled.div`
  width: 40px;
  height: 4px;
  background: rgba(255, 255, 255, 0.3);
  border-radius: 2px;
  overflow: hidden;
  
  &::after {
    content: '';
    display: block;
    height: 100%;
    width: ${props => props.confidence * 100}%;
    background: ${props => props.confidence > 0.7 ? '#4CAF50' :
    props.confidence > 0.4 ? '#FF9800' : '#F44336'};
    transition: width 0.3s ease;
  }
`;

const InputContainer = styled.div`
  padding: 20px;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
`;

const InputWrapper = styled.div`
  display: flex;
  gap: 10px;
  align-items: flex-end;
`;

const TextInput = styled.textarea`
  flex: 1;
  padding: 12px 16px;
  border: none;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.9);
  color: #333;
  font-size: 0.9rem;
  resize: none;
  min-height: 20px;
  max-height: 100px;
  outline: none;
  transition: all 0.3s ease;
  
  &:focus {
    background: white;
    box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.5);
  }
  
  &::placeholder {
    color: #999;
  }
`;

const SendButton = styled.button`
  padding: 12px 20px;
  border: none;
  border-radius: 20px;
  background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
  color: white;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const QuickActions = styled.div`
  display: flex;
  gap: 8px;
  margin-bottom: 10px;
  flex-wrap: wrap;
`;

const QuickActionButton = styled.button`
  padding: 6px 12px;
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 15px;
  background: rgba(255, 255, 255, 0.1);
  color: white;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: translateY(-1px);
  }
`;

const LoadingIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  color: rgba(255, 255, 255, 0.8);
  font-size: 0.9rem;
  
  &::after {
    content: '';
    width: 16px;
    height: 16px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top: 2px solid white;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const MCPChatAssistant = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm your AI trading assistant powered by advanced MCP technology. I can help you with market analysis, trading decisions, and portfolio management. What would you like to know?",
      isUser: false,
      timestamp: new Date(),
      confidence: 0.95,
      type: 'welcome'
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [mcpStatus, setMcpStatus] = useState('connecting');
  const messagesEndRef = useRef(null);

  const quickActions = [
    "Analyze RELIANCE.NS",
    "What's in my portfolio?",
    "Should I buy TCS right now?",
    "How's my risk exposure?",
    "Best stocks to invest in today",
    "Explain my current holdings"
  ];

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    checkMcpStatus();
    const interval = setInterval(checkMcpStatus, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const checkMcpStatus = async () => {
    try {
      const status = await apiService.getMcpStatus();
      setMcpStatus(status.mcp_available && status.server_initialized ? 'connected' : 'disconnected');
    } catch (error) {
      setMcpStatus('disconnected');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isProcessing) return;

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInputValue = inputValue;
    setInputValue('');
    setIsProcessing(true);

    try {
      // Determine if this is a portfolio-related query
      const lowerCaseInput = currentInputValue.toLowerCase();
      const isPortfolioQuery =
        lowerCaseInput.includes('portfolio') ||
        lowerCaseInput.includes('holdings') ||
        lowerCaseInput.includes('positions') ||
        lowerCaseInput.includes('my stocks') ||
        lowerCaseInput.includes('what i own');

      // Determine if this is a trade/buy/sell query
      const isTradeQuery =
        lowerCaseInput.includes('buy') ||
        lowerCaseInput.includes('sell') ||
        lowerCaseInput.includes('trade') ||
        lowerCaseInput.includes('invest') ||
        lowerCaseInput.includes('should i buy') ||
        lowerCaseInput.includes('should i sell');

      // Determine if this is a market analysis query
      const isMarketQuery =
        lowerCaseInput.includes('analyze') ||
        lowerCaseInput.includes('stock') ||
        lowerCaseInput.includes('price') ||
        lowerCaseInput.includes('market') ||
        lowerCaseInput.includes('performance') ||
        lowerCaseInput.includes('trend') ||
        lowerCaseInput.includes('outlook');

      // Determine if this is a risk assessment query
      const isRiskQuery =
        lowerCaseInput.includes('risk') ||
        lowerCaseInput.includes('danger') ||
        lowerCaseInput.includes('safe') ||
        lowerCaseInput.includes('volatility') ||
        lowerCaseInput.includes('drawdown') ||
        lowerCaseInput.includes('protect');

      let response;
      if (mcpStatus === 'connected') {
        // Route to appropriate MCP context based on query type
        let contextType = 'general_trading';

        if (isPortfolioQuery) {
          contextType = 'portfolio_optimization';
        } else if (isTradeQuery) {
          contextType = 'trade_recommendation';
        } else if (isMarketQuery) {
          contextType = 'market_analysis';
        } else if (isRiskQuery) {
          contextType = 'risk_assessment';
        }

        try {
          response = await apiService.mcpChat({
            message: currentInputValue,
            context: { type: contextType }
          });
        } catch (apiError) {
          // Handle API errors specifically
          console.error('API Error:', apiError);

          // Provide more informative error message
          let errorMessage = "I'm having trouble connecting to the analysis engine. ";

          if (apiError.response) {
            switch (apiError.response.status) {
              case 404:
                errorMessage += "The requested resource was not found.";
                break;
              case 500:
                errorMessage += "There was a server error. Please try again later.";
                break;
              case 503:
                errorMessage += "The analysis service is temporarily unavailable.";
                break;
              default:
                errorMessage += `Error code: ${apiError.response.status}`;
            }
          } else if (apiError.request) {
            errorMessage += "Please check your network connection.";
          } else {
            errorMessage += "Please try rephrasing your question.";
          }

          throw new Error(errorMessage);
        }
      } else {
        // Use regular chat
        response = await apiService.sendChatMessage(currentInputValue);
      }

      const aiMessage = {
        id: Date.now() + 1,
        text: response.response || response.message || "I'm here to help with your trading questions!",
        isUser: false,
        timestamp: new Date(),
        confidence: response.confidence || 0.8,
        reasoning: response.reasoning,
        context: response.context,
        portfolioData: response.portfolio_data, // Include portfolio data if available
        errorDetails: response.error_details // Include error details if available
      };

      // Special handling for diagnostic information
      if (response.context === "analysis_error" || response.context === "mcp_scan_failed") {
        aiMessage.text = response.response || "I encountered an issue while analyzing your request. ";
        if (response.error_details) {
          aiMessage.text += `\n\nTechnical details: ${response.error_details}`;
        }
        aiMessage.text += "\n\nThe system is attempting to automatically fetch and process data for new symbols. Please try again in a few moments.";
      }

      // Special handling for cases where no predictions were generated
      if (response.scan_data && response.scan_data.diagnostics) {
        const diagnostics = response.scan_data.diagnostics;

        // Add diagnostic information to the response
        if (diagnostics.issue_summary) {
          aiMessage.text += `\n\nDiagnostic Information:\n${diagnostics.issue_summary}`;
        }

        if (diagnostics.failed_symbols && diagnostics.failed_symbols.length > 0) {
          aiMessage.text += `\n\nSymbols that failed to process: ${diagnostics.failed_symbols.slice(0, 5).join(', ')}`;
          if (diagnostics.failed_symbols.length > 5) {
            aiMessage.text += ` and ${diagnostics.failed_symbols.length - 5} more`;
          }
        }

        if (diagnostics.low_confidence_symbols && diagnostics.low_confidence_symbols.length > 0) {
          aiMessage.text += `\n\nSymbols with low confidence: ${diagnostics.low_confidence_symbols.slice(0, 3).map(s => `${s.symbol} (${s.action}, ${s.confidence.toFixed(2)})`).join(', ')}`;
        }
      }

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Chat error:', error);

      const errorMessage = {
        id: Date.now() + 1,
        text: error.message || "Sorry, I encountered an error processing your request. Please try again.",
        isUser: false,
        timestamp: new Date(),
        isError: true
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleQuickAction = (action) => {
    setInputValue(action);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const getStatusText = () => {
    switch (mcpStatus) {
      case 'connected': return 'MCP AI Connected';
      case 'connecting': return 'Connecting to MCP...';
      case 'disconnected': return 'MCP Disconnected (Basic mode)';
      default: return 'Unknown Status';
    }
  };

  return (
    <ChatContainer>
      <ChatHeader>
        <HeaderTitle>🤖 AI Trading Assistant</HeaderTitle>
        <HeaderSubtitle>Advanced market analysis with MCP technology</HeaderSubtitle>
        <StatusIndicator>
          <StatusDot status={mcpStatus} />
          <StatusText>{getStatusText()}</StatusText>
        </StatusIndicator>
      </ChatHeader>

      <MessagesContainer>
        {messages.map((message) => (
          <Message key={message.id} isUser={message.isUser}>
            <MessageBubble isUser={message.isUser} isError={message.isError}>
              {message.text}
            </MessageBubble>
            {!message.isUser && (
              <MessageMeta>
                <span>{message.timestamp.toLocaleTimeString()}</span>
                {message.confidence !== undefined && (
                  <>
                    <span>•</span>
                    <ConfidenceBar confidence={message.confidence} />
                    <span>{Math.round(message.confidence * 100)}%</span>
                  </>
                )}
                {message.context && (
                  <>
                    <span>•</span>
                    <span>{message.context}</span>
                  </>
                )}
              </MessageMeta>
            )}
          </Message>
        ))}
        {isProcessing && (
          <Message isUser={false}>
            <LoadingIndicator>AI is thinking...</LoadingIndicator>
          </Message>
        )}
        <div ref={messagesEndRef} />
      </MessagesContainer>

      <InputContainer>
        <QuickActions>
          {quickActions.map((action, index) => (
            <QuickActionButton
              key={index}
              onClick={() => handleQuickAction(action)}
              disabled={isProcessing}
            >
              {action}
            </QuickActionButton>
          ))}
        </QuickActions>
        <InputWrapper>
          <TextInput
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me about stocks, market analysis, or trading strategies..."
            disabled={isProcessing}
            rows={1}
          />
          <SendButton
            onClick={handleSubmit}
            disabled={!inputValue.trim() || isProcessing}
          >
            {isProcessing ? '...' : 'Send'}
          </SendButton>
        </InputWrapper>
      </InputContainer>
    </ChatContainer>
  );
};

export default MCPChatAssistant;
