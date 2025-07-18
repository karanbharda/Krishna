// Global variables
const API_BASE_URL = 'http://127.0.0.1:5000/api';

let botData = {
    portfolio: {
        totalValue: 1000000,
        cash: 1000000,
        holdings: {},
        tradeLog: [],
        startingBalance: 1000000
    },
    config: {
        mode: 'paper',
        tickers: [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS',
            'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'AXISBANK.NS', 'LT.NS',
            'HCLTECH.NS', 'WIPRO.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'NESTLEIND.NS'
        ],
        riskLevel: 'MEDIUM',
        maxAllocation: 25
    },
    isRunning: false,
    chatMessages: []
};

let portfolioChart = null;
let allocationChart = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function () {
    initializeApp();
    setupEventListeners();
    updateUI();
    initializeCharts();
    loadWatchlist();
});

function initializeApp() {
    // Initialize session
    botData.sessionId = `session_${new Date().toISOString().replace(/[:.]/g, '')}`;

    // Add welcome message to chat
    if (botData.chatMessages.length === 0) {
        botData.chatMessages.push({
            role: 'assistant',
            content: 'Welcome to the Indian Stock Trading Bot! 🚀\nType a command or ask me anything about trading and markets.',
            timestamp: new Date().toISOString()
        });
    }

    // Load data from backend
    loadDataFromBackend();
}

// API Integration Functions
async function apiCall(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            }
        };

        if (data && method !== 'GET') {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(`${API_BASE_URL}${endpoint}`, options);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error(`API call failed for ${endpoint}:`, error);
        throw error;
    }
}

async function loadDataFromBackend() {
    try {
        // Load portfolio data
        const portfolio = await apiCall('/portfolio');
        botData.portfolio = {
            totalValue: portfolio.total_value,
            cash: portfolio.cash,
            holdings: portfolio.holdings,
            startingBalance: 1000000 // Default starting balance
        };

        // Load recent trades
        const trades = await apiCall('/trades?limit=50');
        botData.portfolio.tradeLog = trades;

        // Load watchlist
        const tickers = await apiCall('/watchlist');
        botData.config.tickers = tickers;

        // Load bot status
        const status = await apiCall('/status');
        botData.isRunning = status.is_running;
        botData.config.mode = status.mode;

        // Update UI with loaded data
        updateUI();

    } catch (error) {
        console.error('Error loading data from backend:', error);
        // Fall back to localStorage or default data
        const savedData = localStorage.getItem('tradingBotData');
        if (savedData) {
            try {
                const parsed = JSON.parse(savedData);
                botData = { ...botData, ...parsed };
            } catch (e) {
                console.error('Error loading saved data:', e);
            }
        }
        updateUI();
    }
}

function setupEventListeners() {
    // Chat input enter key
    document.getElementById('chatInput').addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Ticker input enter key
    document.getElementById('newTickerInput').addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            addTicker();
        }
    });

    // Auto-refresh data every 30 seconds
    setInterval(refreshData, 30000);
}

function updateUI() {
    updatePortfolioMetrics();
    updateRecentActivity();
    updateHoldings();
    updateChatMessages();
    updateModeIndicator();
}

function updatePortfolioMetrics() {
    const metrics = calculateMetrics();

    // Sidebar metrics
    document.getElementById('totalValue').textContent = formatCurrency(metrics.totalValue);
    document.getElementById('cashValue').textContent = formatCurrency(metrics.cash);
    document.getElementById('totalReturn').textContent = formatCurrency(metrics.totalReturn);
    document.getElementById('returnPercentage').textContent = `${metrics.returnPercentage >= 0 ? '+' : ''}${metrics.returnPercentage.toFixed(2)}%`;
    document.getElementById('positionsCount').textContent = Object.keys(botData.portfolio.holdings).length;

    // Dashboard metrics
    document.getElementById('dashboardPortfolioValue').textContent = formatCurrency(metrics.totalValue);
    document.getElementById('unrealizedPnL').textContent = formatCurrency(metrics.unrealizedPnL);
    document.getElementById('activePositions').textContent = Object.keys(botData.portfolio.holdings).length;
    document.getElementById('tradesToday').textContent = getTradesToday();

    // Update return percentage color
    const returnElement = document.getElementById('returnPercentage');
    if (metrics.returnPercentage >= 0) {
        returnElement.className = 'metric-change text-success';
    } else {
        returnElement.className = 'metric-change text-danger';
    }
}

function calculateMetrics() {
    const totalValue = botData.portfolio.totalValue;
    const cash = botData.portfolio.cash;
    const startingBalance = botData.portfolio.startingBalance;
    const totalReturn = totalValue - startingBalance;
    const returnPercentage = (totalReturn / startingBalance) * 100;

    // Calculate unrealized P&L (simplified)
    let unrealizedPnL = 0;
    Object.values(botData.portfolio.holdings).forEach(holding => {
        // In a real implementation, you'd fetch current prices
        unrealizedPnL += (holding.currentPrice || holding.avgPrice) * holding.qty - holding.avgPrice * holding.qty;
    });

    return {
        totalValue,
        cash,
        totalReturn,
        returnPercentage,
        unrealizedPnL
    };
}

function getTradesToday() {
    const today = new Date().toISOString().split('T')[0];
    return botData.portfolio.tradeLog.filter(trade =>
        trade.timestamp.startsWith(today)
    ).length;
}

function updateRecentActivity() {
    const activityContainer = document.getElementById('recentActivity');
    const recentTrades = botData.portfolio.tradeLog.slice(-10).reverse();

    if (recentTrades.length === 0) {
        activityContainer.innerHTML = '<div class="no-activity">No recent trades</div>';
        return;
    }

    activityContainer.innerHTML = recentTrades.map(trade => `
        <div class="activity-item ${trade.action}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>${trade.action.toUpperCase()} ${trade.asset}</strong>
                    <div style="font-size: 0.9rem; color: #666; margin-top: 5px;">
                        ${new Date(trade.timestamp).toLocaleString()}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div><strong>Qty:</strong> ${trade.qty}</div>
                    <div><strong>Price:</strong> ${formatCurrency(trade.price)}</div>
                    <div><strong>Value:</strong> ${formatCurrency(trade.qty * trade.price)}</div>
                </div>
            </div>
        </div>
    `).join('');
}

function updateHoldings() {
    const holdingsContainer = document.getElementById('holdingsTable');
    const holdings = botData.portfolio.holdings;

    if (Object.keys(holdings).length === 0) {
        holdingsContainer.innerHTML = '<div class="no-holdings">No current holdings</div>';
        return;
    }

    const totalValue = calculateMetrics().totalValue;

    holdingsContainer.innerHTML = `
        <div class="holdings-header">
            <div>Ticker</div>
            <div>Quantity</div>
            <div>Avg Price</div>
            <div>Current Value</div>
            <div>% of Portfolio</div>
        </div>
        ${Object.entries(holdings).map(([ticker, data]) => {
        const currentValue = data.qty * data.avgPrice;
        const portfolioPercentage = (currentValue / totalValue * 100).toFixed(1);

        return `
                <div class="holdings-row">
                    <div><strong>${ticker}</strong></div>
                    <div>${data.qty}</div>
                    <div>${formatCurrency(data.avgPrice)}</div>
                    <div>${formatCurrency(currentValue)}</div>
                    <div>${portfolioPercentage}%</div>
                </div>
            `;
    }).join('')}
    `;
}

function updateChatMessages() {
    const chatContainer = document.getElementById('chatMessages');

    chatContainer.innerHTML = botData.chatMessages.map(message => `
        <div class="message ${message.role}">
            <div class="message-content">
                ${message.content.replace(/\n/g, '<br>')}
            </div>
        </div>
    `).join('');

    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function updateModeIndicator() {
    const modeIndicator = document.getElementById('modeIndicator');
    const mode = botData.config.mode;

    if (mode === 'live') {
        modeIndicator.innerHTML = '<div class="mode-badge live-mode">🔴 LIVE TRADING MODE</div>';
    } else {
        modeIndicator.innerHTML = '<div class="mode-badge paper-mode">📝 PAPER TRADING MODE</div>';
    }
}

function loadWatchlist() {
    const watchlistGrid = document.getElementById('watchlistGrid');

    watchlistGrid.innerHTML = botData.config.tickers.map(ticker => `
        <div class="watchlist-item">
            ${ticker}
            <button class="remove-btn" onclick="removeTicker('${ticker}')">×</button>
        </div>
    `).join('');
}

function formatCurrency(amount) {
    return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        minimumFractionDigits: 2
    }).format(amount);
}

// Tab functionality
function showTab(tabName) {
    // Hide all tab panes
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
    });

    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab pane
    document.getElementById(tabName).classList.add('active');

    // Add active class to clicked tab button
    event.target.classList.add('active');

    // Update charts if dashboard tab is selected
    if (tabName === 'dashboard') {
        setTimeout(() => {
            updateCharts();
        }, 100);
    }
}

// Bot control functions
async function startBot() {
    try {
        showLoading();
        await apiCall('/start', 'POST');

        botData.isRunning = true;
        document.getElementById('statusIndicator').innerHTML = `
            <span class="status-dot active"></span>
            <span>Active</span>
        `;

        addChatMessage('assistant', '✅ Trading bot is active and monitoring markets. Check logs for detailed activity.');
        saveData();
    } catch (error) {
        console.error('Error starting bot:', error);
        addChatMessage('assistant', '❌ Error starting bot. Please try again.');
    } finally {
        hideLoading();
    }
}

async function pauseBot() {
    try {
        showLoading();
        await apiCall('/stop', 'POST');

        botData.isRunning = false;
        document.getElementById('statusIndicator').innerHTML = `
            <span class="status-dot" style="background: #f39c12;"></span>
            <span>Paused</span>
        `;

        addChatMessage('assistant', '⏸️ Trading bot has been paused.');
        saveData();
    } catch (error) {
        console.error('Error pausing bot:', error);
        addChatMessage('assistant', '❌ Error pausing bot. Please try again.');
    } finally {
        hideLoading();
    }
}

async function refreshData() {
    try {
        showLoading();
        await loadDataFromBackend();
        console.log('Data refreshed at:', new Date().toLocaleString());
    } catch (error) {
        console.error('Error refreshing data:', error);
        addChatMessage('assistant', '❌ Error refreshing data. Please try again.');
    } finally {
        hideLoading();
    }
}

// Watchlist management
async function addTicker() {
    const input = document.getElementById('newTickerInput');
    const ticker = input.value.trim().toUpperCase();

    if (!ticker) {
        alert('Please enter a ticker symbol');
        return;
    }

    if (botData.config.tickers.includes(ticker)) {
        alert('Ticker already in watchlist');
        return;
    }

    try {
        showLoading();
        const response = await apiCall('/watchlist', 'POST', {
            ticker: ticker,
            action: 'ADD'
        });

        botData.config.tickers = response.tickers;
        input.value = '';
        loadWatchlist();
        saveData();

        addChatMessage('assistant', response.message);
    } catch (error) {
        console.error('Error adding ticker:', error);
        addChatMessage('assistant', `❌ Error adding ${ticker} to watchlist.`);
    } finally {
        hideLoading();
    }
}

async function removeTicker(ticker) {
    try {
        showLoading();
        const response = await apiCall('/watchlist', 'POST', {
            ticker: ticker,
            action: 'REMOVE'
        });

        botData.config.tickers = response.tickers;
        loadWatchlist();
        saveData();

        addChatMessage('assistant', response.message);
    } catch (error) {
        console.error('Error removing ticker:', error);
        addChatMessage('assistant', `❌ Error removing ${ticker} from watchlist.`);
    } finally {
        hideLoading();
    }
}

// Chat functionality
async function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();

    if (!message) return;

    // Add user message
    addChatMessage('user', message);
    input.value = '';

    // Show loading
    showLoading();

    try {
        // Send message to backend
        const response = await apiCall('/chat', 'POST', { message: message });
        addChatMessage('assistant', response.response);

        // Refresh data after command execution
        if (message.startsWith('/')) {
            await loadDataFromBackend();
        }

    } catch (error) {
        console.error('Error sending message:', error);
        // Fall back to local processing
        const response = processCommand(message);
        addChatMessage('assistant', response);
    } finally {
        hideLoading();
        saveData();
    }
}

function addChatMessage(role, content) {
    botData.chatMessages.push({
        role,
        content,
        timestamp: new Date().toISOString()
    });

    updateChatMessages();
}

function processCommand(input) {
    const command = input.toLowerCase().trim();

    // Handle commands
    if (command.startsWith('/')) {
        return handleCommand(command);
    }

    // Handle general queries (simplified AI response)
    return handleGeneralQuery(input);
}

function handleCommand(command) {
    const parts = command.split(' ');
    const cmd = parts[0];

    switch (cmd) {
        case '/start_bot':
            startBot();
            return '✅ Trading bot is active and monitoring markets. Check logs for detailed activity.';

        case '/set_risk':
            const riskLevel = parts[1]?.toUpperCase() || 'MEDIUM';
            if (['LOW', 'MEDIUM', 'HIGH'].includes(riskLevel)) {
                botData.config.riskLevel = riskLevel;
                const stopLoss = { LOW: 3, MEDIUM: 5, HIGH: 8 }[riskLevel];
                return `✅ Risk level set to ${riskLevel}. Stop-loss updated to ${stopLoss}%`;
            }
            return '❌ Invalid risk level. Use: LOW, MEDIUM, or HIGH';

        case '/get_pnl':
            return getPnLReport();

        case '/list_positions':
            return getPositionsReport();

        case '/set_ticker':
            const ticker = parts[1];
            const action = parts[2]?.toUpperCase() || 'ADD';
            return handleTickerCommand(ticker, action);

        case '/pause_trading':
            const minutes = parseInt(parts[1]) || 30;
            pauseBot();
            return `⏸️ Trading paused for ${minutes} minutes`;

        case '/resume_trading':
            startBot();
            return '▶️ Trading resumed successfully.';

        case '/get_performance':
            return getPerformanceReport();

        case '/set_allocation':
            const percentage = parseFloat(parts[1]) || 25;
            if (percentage > 0 && percentage <= 100) {
                botData.config.maxAllocation = percentage;
                return `✅ Maximum allocation per trade set to ${percentage}%`;
            }
            return '❌ Allocation must be between 0 and 100%';

        default:
            return `❌ Unknown command: ${cmd}. Type /help for available commands.`;
    }
}

function handleGeneralQuery(query) {
    // Simplified AI-like responses
    const responses = [
        "I'm here to help with your trading questions! The market is looking interesting today.",
        "Based on current market conditions, it's important to maintain a diversified portfolio.",
        "Remember to always consider risk management in your trading strategy.",
        "The Indian stock market has been showing good momentum lately.",
        "Would you like me to analyze any specific stocks for you?"
    ];

    return responses[Math.floor(Math.random() * responses.length)];
}

function getPnLReport() {
    const metrics = calculateMetrics();

    return `📊 **Portfolio Metrics**
💰 Total Value: ${formatCurrency(metrics.totalValue)}
💵 Cash: ${formatCurrency(metrics.cash)}
📈 Holdings Value: ${formatCurrency(metrics.totalValue - metrics.cash)}
🎯 Total Return: ${formatCurrency(metrics.totalReturn)} (${metrics.returnPercentage >= 0 ? '+' : ''}${metrics.returnPercentage.toFixed(2)}%)
📊 Unrealized P&L: ${formatCurrency(metrics.unrealizedPnL)}
🏢 Active Positions: ${Object.keys(botData.portfolio.holdings).length}`;
}

function getPositionsReport() {
    const holdings = botData.portfolio.holdings;

    if (Object.keys(holdings).length === 0) {
        return '📭 No open positions currently.';
    }

    let report = '📊 **Current Positions:**\n';
    Object.entries(holdings).forEach(([ticker, data]) => {
        const currentValue = data.qty * data.avgPrice;
        report += `• ${ticker}: ${data.qty} shares @ ${formatCurrency(data.avgPrice)} (${formatCurrency(currentValue)})\n`;
    });

    return report;
}

function handleTickerCommand(ticker, action) {
    if (!ticker) {
        return '❌ Please specify a ticker. Example: /set_ticker RELIANCE.NS ADD';
    }

    if (action === 'ADD') {
        if (!botData.config.tickers.includes(ticker)) {
            botData.config.tickers.push(ticker);
            loadWatchlist();
            return `✅ Added ${ticker} to watchlist. Total tickers: ${botData.config.tickers.length}`;
        } else {
            return `ℹ️ ${ticker} is already in watchlist.`;
        }
    } else if (action === 'REMOVE') {
        const index = botData.config.tickers.indexOf(ticker);
        if (index > -1) {
            botData.config.tickers.splice(index, 1);
            loadWatchlist();
            return `✅ Removed ${ticker} from watchlist. Total tickers: ${botData.config.tickers.length}`;
        } else {
            return `ℹ️ ${ticker} is not in watchlist.`;
        }
    }

    return '❌ Invalid action. Use ADD or REMOVE.';
}

function getPerformanceReport() {
    const metrics = calculateMetrics();
    const tradesCount = botData.portfolio.tradeLog.length;

    return `📈 **Performance Report**
💰 Current Value: ${formatCurrency(metrics.totalValue)}
🎯 Total Return: ${formatCurrency(metrics.totalReturn)} (${metrics.returnPercentage >= 0 ? '+' : ''}${metrics.returnPercentage.toFixed(2)}%)
📊 Unrealized P&L: ${formatCurrency(metrics.unrealizedPnL)}
🔄 Total Trades: ${tradesCount}
🏢 Active Positions: ${Object.keys(botData.portfolio.holdings).length}`;
}

function toggleCommandHelp() {
    const helpDiv = document.getElementById('commandHelp');
    helpDiv.classList.toggle('hidden');
}

// Settings functionality
async function openSettings() {
    try {
        // Load current settings from backend
        const settings = await apiCall('/settings');

        document.getElementById('settingsModal').classList.remove('hidden');

        // Populate current settings
        document.getElementById('tradingMode').value = settings.mode || botData.config.mode;
        document.getElementById('riskLevel').value = botData.config.riskLevel;
        document.getElementById('maxAllocation').value = (settings.max_capital_per_trade * 100) || botData.config.maxAllocation;
    } catch (error) {
        console.error('Error loading settings:', error);
        document.getElementById('settingsModal').classList.remove('hidden');

        // Use local settings as fallback
        document.getElementById('tradingMode').value = botData.config.mode;
        document.getElementById('riskLevel').value = botData.config.riskLevel;
        document.getElementById('maxAllocation').value = botData.config.maxAllocation;
    }
}

function closeSettings() {
    document.getElementById('settingsModal').classList.add('hidden');
}

async function saveSettings() {
    try {
        showLoading();

        const newSettings = {
            mode: document.getElementById('tradingMode').value,
            max_capital_per_trade: parseFloat(document.getElementById('maxAllocation').value) / 100,
            stop_loss_pct: { LOW: 0.03, MEDIUM: 0.05, HIGH: 0.08 }[document.getElementById('riskLevel').value] || 0.05
        };

        await apiCall('/settings', 'POST', newSettings);

        // Update local config
        botData.config.mode = newSettings.mode;
        botData.config.riskLevel = document.getElementById('riskLevel').value;
        botData.config.maxAllocation = parseFloat(document.getElementById('maxAllocation').value);

        updateModeIndicator();
        saveData();
        closeSettings();

        addChatMessage('assistant', '✅ Settings saved successfully!');
    } catch (error) {
        console.error('Error saving settings:', error);
        addChatMessage('assistant', '❌ Error saving settings. Please try again.');
    } finally {
        hideLoading();
    }
}

// Loading overlay
function showLoading() {
    document.getElementById('loadingOverlay').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.add('hidden');
}

// Data persistence
function saveData() {
    try {
        localStorage.setItem('tradingBotData', JSON.stringify(botData));
    } catch (e) {
        console.error('Error saving data:', e);
    }
}

// Chart initialization and updates
function initializeCharts() {
    initializePortfolioChart();
    initializeAllocationChart();
}

function initializePortfolioChart() {
    const ctx = document.getElementById('portfolioChart').getContext('2d');

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

    portfolioChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Portfolio Value',
                data: data,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function (value) {
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
        }
    });
}

function initializeAllocationChart() {
    const ctx = document.getElementById('allocationChart').getContext('2d');

    // Sample allocation data
    const holdings = botData.portfolio.holdings;
    const labels = Object.keys(holdings).length > 0 ? Object.keys(holdings) : ['Cash'];
    const data = Object.keys(holdings).length > 0
        ? Object.values(holdings).map(h => h.qty * h.avgPrice)
        : [botData.portfolio.cash];

    allocationChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: [
                    '#3498db', '#e74c3c', '#f39c12', '#27ae60', '#9b59b6',
                    '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#2ecc71'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function updateCharts() {
    if (portfolioChart) {
        portfolioChart.update();
    }
    if (allocationChart) {
        // Update allocation chart with current holdings
        const holdings = botData.portfolio.holdings;
        const labels = Object.keys(holdings).length > 0 ? Object.keys(holdings) : ['Cash'];
        const data = Object.keys(holdings).length > 0
            ? Object.values(holdings).map(h => h.qty * h.avgPrice)
            : [botData.portfolio.cash];

        allocationChart.data.labels = labels;
        allocationChart.data.datasets[0].data = data;
        allocationChart.update();
    }
}
