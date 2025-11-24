# Execution Flow Diagram

```mermaid
graph TB
    A[MCP Predictions] --> B[Risk Engine]
    C[News/Sentiment Analysis] --> B
    B --> D[Position Sizing]
    D --> E[Dhan Execution]
    E --> F[Trade Logs]
    
    B --> G[Volatility Filter]
    B --> H[Stoploss/Takeprofit]
    B --> I[Max Position Size]
    B --> J[Sentiment Conflict Filter]
    
    D --> K[Sentiment Multiplier]
    D --> L[News Confidence Override]
    
    E --> M[Autonomous Mode]
    E --> N[Approval Mode]
    
    O[UI/API] --> P[/trade/preview]
    O --> Q[/action/final_decision]
    O --> R[/trade/execute]
    O --> S[/trade/logs]
    
    P --> D
    Q --> B
    R --> E
    S --> F
```

## Component Flow Description

1. **MCP Predictions**: Generates trading signals using machine learning models
2. **Risk Engine**: Applies risk management rules and filters
3. **Position Sizing**: Calculates optimal position size with sentiment adjustments
4. **Dhan Execution**: Executes trades through Dhan API with order management
5. **Trade Logs**: Records all executed trades for audit and analysis

## Key Features Implemented

- **Final Filters**: Stoploss/takeprofit, max position size, volatility filter, sentiment conflict filter
- **Sentiment Integration**: Sentiment multiplier and news-confidence override
- **Mode Toggling**: Autonomous mode (immediate execution) and Approval mode (manual approval)
- **API Endpoints**: Preview, final decision, execute, and logs endpoints
- **Comprehensive Logging**: Trade execution tracking and monitoring