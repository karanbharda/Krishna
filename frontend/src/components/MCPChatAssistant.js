import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import apiService from '../services/apiService';

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
  const [isLoading, setIsLoading] = useState(false);
  const [mcpStatus, setMcpStatus] = useState('connecting');
  const messagesEndRef = useRef(null);

  const quickActions = [
    "Analyze RELIANCE.NS",
    "Portfolio risk assessment", 
    "Market outlook today",
    "Best stocks to buy",
    "Trading strategy advice"
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

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Determine if this is a market analysis request
      const isMarketQuery = inputValue.toLowerCase().includes('analyze') || 
                           inputValue.toLowerCase().includes('stock') ||
                           inputValue.toLowerCase().includes('price');

      let response;
      if (isMarketQuery && mcpStatus === 'connected') {
        // Use MCP for market analysis
        response = await apiService.mcpChat({
          message: inputValue,
          context: { type: 'market_analysis' }
        });
      } else {
        // Use regular chat
        response = await apiService.sendChatMessage(inputValue);
      }

      const aiMessage = {
        id: Date.now() + 1,
        text: response.response || response.message || "I'm here to help with your trading questions!",
        isUser: false,
        timestamp: new Date(),
        confidence: response.confidence || 0.8,
        reasoning: response.reasoning,
        context: response.context
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = {
        id: Date.now() + 1,
        text: "I'm sorry, I encountered an error. Please try again or check if the MCP server is running.",
        isUser: false,
        timestamp: new Date(),
        confidence: 0.0,
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickAction = (action) => {
    setInputValue(action);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
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
        {isLoading && (
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
              disabled={isLoading}
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
            disabled={isLoading}
            rows={1}
          />
          <SendButton 
            onClick={handleSendMessage} 
            disabled={!inputValue.trim() || isLoading}
          >
            {isLoading ? '...' : 'Send'}
          </SendButton>
        </InputWrapper>
      </InputContainer>
    </ChatContainer>
  );
};

export default MCPChatAssistant;
