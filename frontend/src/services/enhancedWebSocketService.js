/**
 * Phase 1: Enhanced WebSocket Service
 * Replaces 5-second polling with real-time data streaming
 */

class EnhancedWebSocketService {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.subscriptions = new Map();
        this.messageQueue = [];
        this.heartbeatInterval = null;
        this.connectionPromise = null;

        // Event listeners
        this.onPortfolioUpdate = null;
        this.onTradeUpdate = null;
        this.onMarketDataUpdate = null;
        this.onSignalUpdate = null;
        this.onConnectionChange = null;
        this.onError = null;

        console.log('✅ Enhanced WebSocket Service initialized');
    }

    /**
     * Connect to WebSocket server
     */
    async connect() {
        if (this.connectionPromise) {
            return this.connectionPromise;
        }

        this.connectionPromise = new Promise((resolve, reject) => {
            try {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const host = process.env.NODE_ENV === 'production'
                    ? window.location.host
                    : 'localhost:5000';

                const wsUrl = `${protocol}//${host}/ws`;
                console.log(`🔌 Connecting to WebSocket: ${wsUrl}`);

                this.ws = new WebSocket(wsUrl);

                this.ws.onopen = () => {
                    console.log('✅ WebSocket connected');
                    this.isConnected = true;
                    this.reconnectAttempts = 0;
                    this.connectionPromise = null;

                    // Process queued messages
                    this._processMessageQueue();

                    // Start heartbeat
                    this._startHeartbeat();

                    // Subscribe to default updates
                    this.subscribe(['portfolio', 'trades', 'market_data', 'heartbeat']);

                    // Request initial data
                    this.send('get_initial_data');

                    if (this.onConnectionChange) {
                        this.onConnectionChange(true);
                    }

                    resolve();
                };

                this.ws.onmessage = (event) => {
                    this._handleMessage(event.data);
                };

                this.ws.onclose = (event) => {
                    console.log('❌ WebSocket disconnected:', event.code, event.reason);
                    this.isConnected = false;
                    this.connectionPromise = null;
                    this._stopHeartbeat();

                    if (this.onConnectionChange) {
                        this.onConnectionChange(false);
                    }

                    // Auto-reconnect
                    this._scheduleReconnect();
                };

                this.ws.onerror = (error) => {
                    console.error('❌ WebSocket error:', error);
                    if (this.onError) {
                        this.onError(error);
                    }
                    reject(error);
                };

            } catch (error) {
                console.error('❌ Failed to create WebSocket:', error);
                this.connectionPromise = null;
                reject(error);
            }
        });

        return this.connectionPromise;
    }

    /**
     * Disconnect from WebSocket
     */
    disconnect() {
        if (this.ws) {
            this.ws.close(1000, 'Manual disconnect');
            this.ws = null;
        }
        this.isConnected = false;
        this._stopHeartbeat();
        console.log('🔌 WebSocket disconnected manually');
    }

    /**
     * Send message to server
     */
    send(message) {
        if (this.isConnected && this.ws) {
            this.ws.send(message);
        } else {
            // Queue message for later
            this.messageQueue.push(message);

            // Try to connect if not connected
            if (!this.isConnected) {
                this.connect().catch(console.error);
            }
        }
    }

    /**
     * Subscribe to specific update types
     */
    subscribe(updateTypes) {
        const subscribeMessage = `subscribe:${updateTypes.join(',')}`;
        this.send(subscribeMessage);
    }

    /**
     * Register event listeners
     */
    on(event, callback) {
        switch (event) {
            case 'portfolioUpdate':
                this.onPortfolioUpdate = callback;
                break;
            case 'tradeUpdate':
                this.onTradeUpdate = callback;
                break;
            case 'marketDataUpdate':
                this.onMarketDataUpdate = callback;
                break;
            case 'signalUpdate':
                this.onSignalUpdate = callback;
                break;
            case 'connectionChange':
                this.onConnectionChange = callback;
                break;
            case 'error':
                this.onError = callback;
                break;
            default:
                console.warn(`Unknown event type: ${event}`);
        }
    }

    /**
     * Handle incoming WebSocket messages
     */
    _handleMessage(data) {
        try {
            const message = JSON.parse(data);

            // Handle pong response
            if (data === 'pong') {
                return;
            }

            switch (message.type) {
                case 'initial_data':
                    // Broadcast initial data to all listeners
                    if (this.onPortfolioUpdate && message.data.portfolio) {
                        this.onPortfolioUpdate(message.data.portfolio);
                    }
                    if (this.onTradeUpdate && message.data.trades) {
                        this.onTradeUpdate(message.data.trades);
                    }
                    if (this.onMarketDataUpdate && message.data.market_data) {
                        this.onMarketDataUpdate(message.data.market_data);
                    }
                    console.log('📊 Initial data received');
                    break;

                case 'portfolio':
                    if (this.onPortfolioUpdate) {
                        this.onPortfolioUpdate(message.data);
                    }
                    break;

                case 'trades':
                    if (this.onTradeUpdate) {
                        this.onTradeUpdate(message.data);
                    }
                    break;

                case 'market_data':
                    if (this.onMarketDataUpdate) {
                        this.onMarketDataUpdate(message.data);
                    }
                    break;

                case 'signals':
                    if (this.onSignalUpdate) {
                        this.onSignalUpdate(message.data);
                    }
                    break;

                case 'heartbeat':
                    // Handle heartbeat
                    console.log('💓 Heartbeat received');
                    break;

                case 'error':
                    console.error('❌ Server error:', message.data);
                    if (this.onError) {
                        this.onError(message.data);
                    }
                    break;

                default:
                    console.log('📨 Unknown message type:', message.type, message.data);
            }
        } catch (error) {
            console.error('❌ Error parsing WebSocket message:', error, data);
        }
    }

    /**
     * Process queued messages
     */
    _processMessageQueue() {
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            this.send(message);
        }
    }

    /**
     * Start heartbeat to keep connection alive
     */
    _startHeartbeat() {
        this._stopHeartbeat(); // Clear any existing interval

        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected) {
                this.send('ping');
            }
        }, 30000); // Ping every 30 seconds
    }

    /**
     * Stop heartbeat
     */
    _stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    /**
     * Schedule reconnection attempt
     */
    _scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('❌ Max reconnection attempts reached');
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

        console.log(`🔄 Scheduling reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`);

        setTimeout(() => {
            if (!this.isConnected) {
                this.connect().catch(console.error);
            }
        }, delay);
    }

    /**
     * Get connection status
     */
    getConnectionStatus() {
        return {
            isConnected: this.isConnected,
            reconnectAttempts: this.reconnectAttempts,
            maxReconnectAttempts: this.maxReconnectAttempts,
            hasListeners: {
                portfolio: !!this.onPortfolioUpdate,
                trades: !!this.onTradeUpdate,
                marketData: !!this.onMarketDataUpdate,
                signals: !!this.onSignalUpdate
            }
        };
    }
}

// Create singleton instance
const enhancedWebSocketService = new EnhancedWebSocketService();

// Auto-connect when service is imported
enhancedWebSocketService.connect().catch(error => {
    console.warn('⚠️ Initial WebSocket connection failed:', error);
});

export default enhancedWebSocketService; 