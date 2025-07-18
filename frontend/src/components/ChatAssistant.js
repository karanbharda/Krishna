import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';

const ChatContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 600px;
`;

const ChatHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 2px solid #e9ecef;

  h3 {
    margin: 0;
    color: #2c3e50;
  }
`;

const HelpButton = styled.button`
  background: #f39c12;
  color: white;
  border: none;
  padding: 8px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.3s ease;

  &:hover {
    background: #e67e22;
  }
`;

const CommandHelp = styled.div`
  background: #f8f9fa;
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 15px;
  border: 1px solid #e9ecef;
  display: ${props => props.show ? 'block' : 'none'};

  h4 {
    color: #2c3e50;
    margin-bottom: 10px;
  }
`;

const CommandsGrid = styled.div`
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 20px;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const CommandGroup = styled.div`
  h5 {
    color: #2c3e50;
    margin-bottom: 10px;
  }

  ul {
    list-style: none;
    padding-left: 0;
  }

  li {
    margin-bottom: 5px;
    font-size: 0.9rem;
  }

  code {
    background: #e9ecef;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
    color: #e74c3c;
  }
`;

const ChatMessages = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
  margin-bottom: 15px;
  border: 1px solid #e9ecef;
`;

const Message = styled.div`
  margin-bottom: 15px;
  display: flex;
  align-items: flex-start;
  gap: 10px;
  flex-direction: ${props => props.isUser ? 'row-reverse' : 'row'};
`;

const MessageContent = styled.div`
  background: ${props => props.isUser ? '#3498db' : '#ecf0f1'};
  color: ${props => props.isUser ? 'white' : '#2c3e50'};
  padding: 12px 15px;
  border-radius: 15px;
  max-width: 70%;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  line-height: 1.4;
  white-space: pre-wrap;
  word-wrap: break-word;
`;

const MessageTime = styled.div`
  font-size: 0.8rem;
  color: #7f8c8d;
  margin-top: 5px;
  text-align: ${props => props.isUser ? 'right' : 'left'};
`;

const ChatInputContainer = styled.div`
  display: flex;
  gap: 10px;
`;

const ChatInput = styled.input`
  flex: 1;
  padding: 12px;
  border: 2px solid #e9ecef;
  border-radius: 25px;
  font-size: 1rem;
  outline: none;

  &:focus {
    border-color: #3498db;
  }

  &:disabled {
    background: #f8f9fa;
    cursor: not-allowed;
  }
`;

const SendButton = styled.button`
  background: #3498db;
  color: white;
  border: none;
  padding: 12px 15px;
  border-radius: 50%;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;

  &:hover:not(:disabled) {
    background: #2980b9;
    transform: scale(1.1);
  }

  &:disabled {
    background: #95a5a6;
    cursor: not-allowed;
    transform: none;
  }
`;

const WelcomeMessage = styled.div`
  text-align: center;
  color: #7f8c8d;
  font-style: italic;
  padding: 20px;
`;

const ChatAssistant = ({ messages, onSendMessage }) => {
  const [inputMessage, setInputMessage] = useState('');
  const [showHelp, setShowHelp] = useState(false);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || loading) return;

    const message = inputMessage.trim();
    setInputMessage('');
    setLoading(true);

    try {
      await onSendMessage(message);
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  return (
    <ChatContainer>
      <ChatHeader>
        <h3>🤖 Trading Assistant</h3>
        <HelpButton onClick={() => setShowHelp(!showHelp)}>
          <i className="fas fa-question-circle"></i> Commands
        </HelpButton>
      </ChatHeader>

      <CommandHelp show={showHelp}>
        <h4>Available Commands:</h4>
        <CommandsGrid>
          <CommandGroup>
            <h5>Trading Commands:</h5>
            <ul>
              <li><code>/start_bot</code> - Verify bot status</li>
              <li><code>/set_risk HIGH</code> - Set risk level (LOW/MEDIUM/HIGH)</li>
              <li><code>/get_pnl</code> - Get portfolio metrics</li>
              <li><code>/why_trade SBIN.NS</code> - Get trade analysis</li>
              <li><code>/list_positions</code> - List open positions</li>
              <li><code>/set_ticker RELIANCE.NS ADD</code> - Add/remove tickers</li>
              <li><code>/get_signals TCS.NS</code> - Get trading signals</li>
              <li><code>/pause_trading 30</code> - Pause trading for 30 minutes</li>
              <li><code>/resume_trading</code> - Resume trading</li>
              <li><code>/get_performance 1w</code> - Get performance report</li>
              <li><code>/set_allocation 20</code> - Set max allocation per trade</li>
            </ul>
          </CommandGroup>
          <CommandGroup>
            <h5>General Chat:</h5>
            <p>You can also ask general questions about trading, markets, or your portfolio!</p>
          </CommandGroup>
        </CommandsGrid>
      </CommandHelp>

      <ChatMessages>
        {messages.length === 0 ? (
          <WelcomeMessage>
            Welcome to the Indian Stock Trading Bot! 🚀<br />
            Type a command or ask me anything about trading and markets.
          </WelcomeMessage>
        ) : (
          messages.map((message, index) => (
            <Message key={index} isUser={message.role === 'user'}>
              <div>
                <MessageContent isUser={message.role === 'user'}>
                  {message.content}
                </MessageContent>
                <MessageTime isUser={message.role === 'user'}>
                  {formatTimestamp(message.timestamp)}
                </MessageTime>
              </div>
            </Message>
          ))
        )}
        <div ref={messagesEndRef} />
      </ChatMessages>

      <ChatInputContainer>
        <ChatInput
          ref={inputRef}
          type="text"
          placeholder="Ask a question or enter a command..."
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={loading}
        />
        <SendButton 
          onClick={handleSendMessage}
          disabled={loading || !inputMessage.trim()}
        >
          {loading ? (
            <div className="loading-spinner"></div>
          ) : (
            <i className="fas fa-paper-plane"></i>
          )}
        </SendButton>
      </ChatInputContainer>
    </ChatContainer>
  );
};

export default ChatAssistant;
