from datetime import timedelta
import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State # type: ignore
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from flask_cors import CORS 


app = dash.Dash(__name__)
app.title = "Cryptocurrency Trading Signals Dashboard"
server = app.server

CORS(server, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],  # Your Next.js frontend URL
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Create the dropdown dictionary for easy lookup
dropdown_dict = {
    'BTC': 'Bitcoin',
    'ETH': 'Ethereum',
    'BNB': 'BNB',
    'SOL': 'Solana',
    'DOGE': 'Dogecoin',
    'ADA': 'Cardano',
    'TRX': 'TRON',
    'LINK': 'Chainlink',
    'XLM': 'Stellar',
    'AVAX': 'Avalanche',
    'LTC': 'Litecoin',
    'TON': 'Toncoin',
    'SHIB': 'Shiba Inu',
    'LEO': 'UNUS SED LEO',
    'HBAR': 'Hedera',
    'DOT': 'Polkadot',
    'BCH': 'Bitcoin Cash',
    'UNI': 'Uniswap',
    'XMR': 'Monero',
    'PEPE': 'Pepe',
    'AAVE': 'Aave',
    'NEAR': 'NEAR Protocol',
    'ONDO': 'Ondo finance',
    'MNT': 'Mantle',
    'TRUMP': 'OFFICIAL TRUMP',
    'APT': 'Aptos',
    'ETC': 'Ethereum Classic',
    'TAO': 'Bittensor',
    'ICP': 'Internet Computer',
    'OKB': 'OKB',
    'VET': 'VeChain',
    'POL': 'polygon-ecosystem-token',
    'KAS': 'Kaspa',
    'CRO': 'Cronos',
    'ALGO': 'Algorand',
    'JUP': 'Jupiter-ag',
    'RENDER': 'Render',
    'ARB': 'Arbitrum',
    'GT': 'GateToken',
    'FIL': 'Filecoin',
    'LDO': 'Lido DAO',
    'OP': 'optimism-ethereum',
    'TIA': 'Celestia',
    'FET': 'Artificial Superintelligence Alliance',
    'ATOM': 'Cosmos',
    'KCS': 'KuCoin Token',
    'INJ': 'Injective',
    'XDC': 'XDC Network',
    'DEXE': 'DeXe',
    'ENA': 'Ethena',
    'STX': 'Stacks',
    'FLR': 'Flare',
    'BONK': 'Bonk1',
    'GRT': 'The Graph',
    'RAY': 'Raydium',
    'QNT': 'Quant',
    'IMX': 'immutable-x',
    'SEI': 'Sei',
    'THETA': 'Theta Network',
    'WLD': 'worldcoin-org',
    'MOVE': 'Movement',
    'JASMY': 'Jasmy',
    'PYTH': 'Pyth Network',
    'BSV': 'Bitcoin SV',
    'RON': 'Ronin',
    'ENS': 'Ethereum Name Service',
    'FLOKI': 'FLOKI-inu',
    'SAND': 'The Sandbox',
    'NEO': 'Neo',
    'KAIA': 'Kaia',
    'JTO': 'Jito',
    'NEXO': 'Nexo',
    'GALA': 'Gala',
    'IOTA': 'IOTA',
    'FLOW': 'Flow',
    'BTT': 'bittorrent-new',
    'XTZ': 'Tezos',
    'EOS': 'EOS',
    'MKR': 'Maker',
    'AR': 'Arweave',
    'MELANIA': 'Melania Meme',
    'WIF': 'dogwifhat',
    'AERO': 'Aerodrome Finance',
    'AXS': 'Axie Infinity',
    'CRV': 'Curve DAO Token',
    'AIOZ': 'AIOZ Network',
    'PENGU': 'Pudgy Penguins',
    'STRK': 'Starknet-token',
    'MANA': 'Decentraland',
    'SPX': 'SPX6900',
    'EGLD': 'MultiversX-egld',
    'XCN': 'Onyxcoin',
    'FARTCOIN': 'Fartcoin',
    'VIRTUAL': 'Virtual Protocol',
    'AVAX_USD': 'AVAX_USD CoinW',
    'PI_USD': 'PI_USD OKX',
    'TRX_USD': 'TRX_USD Binance'
}

# Create the stock_list from the stock_map
stock_list = [{'label': value, 'value': key} for key, value in dropdown_dict.items()]

def load_crypto_data(crypto_symbol):
    """Load and preprocess cryptocurrency data based on symbol"""
    try:
        # Handle special cases for USD pairs
        if crypto_symbol in ['AVAX_USD', 'PI_USD', 'TRX_USD']:
            filename = f"./Data/{dropdown_dict[crypto_symbol]} Historical Data.csv"
        else:
            # For regular cryptocurrency files
            filename = f"./Data/{dropdown_dict[crypto_symbol]}_{crypto_symbol}_historical_data.csv"
       
        df = pd.read_csv(filename)
        df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
        df.set_index('Date', inplace=True)

        # Clean numeric columns by removing $ and , characters
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].str.replace('$', '').str.replace(',', '').astype(float)
       
        return df
    except Exception as e:
        print(f"Error loading data for {crypto_symbol}: {str(e)}")
        return None

# Initialize with Bitcoin data
df = load_crypto_data('BTC')

def SMA(df, period=50, column="Close"):
    return df[column].rolling(window=period).mean()

def EMA(df, period=50, column="Close"):
    return df[column].ewm(span=period, adjust=False).mean()

def WMA(df, period=50, column="Close"):
    weights = np.arange(1, period + 1)
    return df[column].rolling(window=period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

def VWMA(df, period=50, column="Close"):
    volume = df["Volume"]
    return (df[column] * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()

def MA(df, period=30, column="Close", ma_type="SMA"):
    if ma_type == "SMA":
        return SMA(df, period, column)
    elif ma_type == "EMA":
        return EMA(df, period, column)
    elif ma_type == "WMA":
        return WMA(df, period, column)
    elif ma_type == "VWMA":
        return VWMA(df, period, column)
    else:
        raise ValueError("Invalid ma_type. Use 'SMA', 'EMA', 'WMA', or 'VWMA'.")

def buy_n_sell(data, crypto_symbol='BTC', period1=20, period2=50, period3=200, MA_type='SMA', show_bollinger=False):
    try:
        df = data.copy()
        df['line1'] = MA(df, period=period1, column="Close", ma_type=MA_type)
        df['line2'] = MA(df, period=period2, column="Close", ma_type=MA_type)
        df['line3'] = MA(df, period=period3, column="Close", ma_type=MA_type)

        # Condition 1
        df['Signal'] = np.where(df["line1"] > df["line2"], 1, 0)
        df['Position'] = df['Signal'].diff()
        df['Buy'] = np.where(df['Position'] == 1, df["Close"], np.nan)
        df['Sell'] = np.where(df['Position'] == -1, df["Close"], np.nan)

        # Condition 2
        df['Golden_Signal'] = np.where(df["line2"] > df["line3"], 1, 0)
        df['Golden_Position'] = df['Golden_Signal'].diff()
        df['Golden_Buy'] = np.where(df['Golden_Position'] == 1, df["Close"], np.nan)
        df['Death_Sell'] = np.where(df['Golden_Position'] == -1, df["Close"], np.nan)

        # Create candlestick chart and buy/sell signals
        fig = go.Figure()
        tv_colors = {
            'background': '#131722',
            'grid': '#2A2E39',
            'text': '#D9D9D9',
            'candle_up': '#26A69A',
            'candle_down': '#EF5350',
            'wick_up': '#26A69A',
            'wick_down': '#EF5350',
            'ma_short': '#2196F3',
            'ma_medium': '#FF9800',
            'ma_long': '#4CAF50',
            'buy': '#00E676',
            'sell': '#FF5252',
            'golden_buy': '#FFD700',
            'death_sell': '#8B0000',
            'volume_up': 'rgba(38, 166, 154, 0.3)',
            'volume_down': 'rgba(239, 83, 80, 0.3)',
            'support': '#00FF00',
            'resistance': '#FF0000'
        }
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color=tv_colors['candle_up'],
            decreasing_line_color=tv_colors['candle_down'],
            increasing_fillcolor=tv_colors['candle_up'],
            decreasing_fillcolor=tv_colors['candle_down']
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['line1'],
            mode='lines',
            line=dict(color=tv_colors['ma_short'], width=1),
            name=f'{MA_type} {period1}'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['line2'],
            mode='lines',
            line=dict(color=tv_colors['ma_medium'], width=1),
            name=f'{MA_type} {period2}'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['line3'],
            mode='lines',
            line=dict(color=tv_colors['ma_long'], width=1),
            name=f'{MA_type} {period3}'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Buy'],
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color=tv_colors['buy']),
            name='Buy Signal'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Sell'],
            mode='markers',
            marker=dict(symbol='triangle-down', size=10, color=tv_colors['sell']),
            name='Sell Signal'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Golden_Buy'],
            mode='markers',
            marker=dict(symbol='star', size=12, color=tv_colors['death_sell']),
            name='Death sell'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Death_Sell'],
            mode='markers',
            marker=dict(symbol='star', size=12, color=tv_colors['golden_buy']),
            name='Golden buy'
        ))
        fig.update_layout(
            title=f'{crypto_symbol} Price Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            dragmode='zoom',
            modebar_remove=['drawline', 'drawopenpath', 'eraseshape', 'lasso2d', 'select2d'],
            xaxis=dict(
                rangeslider=dict(visible=False),
                type='date',
                autorange=True,
                fixedrange=False
            ),
            yaxis=dict(
                autorange=True,
                fixedrange=False
            )
        )
        return fig
    except Exception as e:
        print(f"Error calculating buy/sell signals: {str(e)}")
        return go.Figure()

def RSI(df, period=14, column="Close"):
    """Calculate Relative Strength Index"""
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def MACD(df, fast_period=12, slow_period=26, signal_period=9, column="Close"):
    """Calculate MACD, Signal Line, and MACD Histogram"""
    # Calculate the MACD line
    exp1 = df[column].ewm(span=fast_period, adjust=False).mean()
    exp2 = df[column].ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
   
    # Calculate the Signal line
    signal = macd.ewm(span=signal_period, adjust=False).mean()
   
    # Calculate the MACD histogram
    hist = macd - signal
   
    return macd, signal, hist

def create_rsi_chart(df, period=14):
    """Create RSI chart with overbought/oversold levels"""
    rsi = RSI(df, period)
   
    fig = go.Figure()
   
    # Add RSI line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=rsi,
        name='RSI',
        line=dict(color='#2196F3', width=1)
    ))
   
    # Add overbought level
    fig.add_hline(y=70, line_dash="dash", line_color="#EF5350", annotation_text="Overbought")
   
    # Add oversold level
    fig.add_hline(y=30, line_dash="dash", line_color="#26A69A", annotation_text="Oversold")
   
    # Update layout with TradingView-like styling
    fig.update_layout(
        title=dict(
            text=f'<b>RSI ({period})</b>',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(color='#D9D9D9')
        ),
        xaxis=dict(
            title='Date',
            gridcolor='#2A2E39',
            zerolinecolor='#2A2E39',
            showgrid=True,
            rangeslider=dict(visible=False),
            type='date',
            tickfont=dict(color='#D9D9D9'),
            titlefont=dict(color='#D9D9D9')
        ),
        yaxis=dict(
            title='RSI',
            gridcolor='#2A2E39',
            zerolinecolor='#2A2E39',
            showgrid=True,
            range=[0, 100],
            tickfont=dict(color='#D9D9D9'),
            titlefont=dict(color='#D9D9D9')
        ),
        plot_bgcolor='#131722',
        paper_bgcolor='#131722',
        font=dict(color='#D9D9D9'),
        height=200,
        margin=dict(t=50, b=50, l=50, r=50),
        showlegend=False
    )
   
    return fig

def create_macd_chart(df, fast_period=12, slow_period=26, signal_period=9):
    """Create MACD chart with signal line and histogram"""
    macd, signal, hist = MACD(df, fast_period, slow_period, signal_period)
   
    fig = go.Figure()
   
    # Add MACD line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=macd,
        name='MACD',
        line=dict(color='#2196F3', width=1)
    ))
   
    # Add Signal line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=signal,
        name='Signal',
        line=dict(color='#FF9800', width=1)
    ))
   
    # Add Histogram
    fig.add_trace(go.Bar(
        x=df.index,
        y=hist,
        name='Histogram',
        marker_color=np.where(hist >= 0, '#26A69A', '#EF5350'),
        opacity=0.5
    ))
   
    # Update layout with TradingView-like styling
    fig.update_layout(
        title=dict(
            text=f'<b>MACD ({fast_period}, {slow_period}, {signal_period})</b>',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(color='#D9D9D9')
        ),
        xaxis=dict(
            title='Date',
            gridcolor='#2A2E39',
            zerolinecolor='#2A2E39',
            showgrid=True,
            rangeslider=dict(visible=False),
            type='date',
            tickfont=dict(color='#D9D9D9'),
            titlefont=dict(color='#D9D9D9')
        ),
        yaxis=dict(
            title='MACD',
            gridcolor='#2A2E39',
            zerolinecolor='#2A2E39',
            showgrid=True,
            tickfont=dict(color='#D9D9D9'),
            titlefont=dict(color='#D9D9D9')
        ),
        plot_bgcolor='#131722',
        paper_bgcolor='#131722',
        font=dict(color='#D9D9D9'),
        height=200,
        margin=dict(t=50, b=50, l=50, r=50),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#D9D9D9')
        )
    )
   
    return fig

def ATR(df, period=14):
    """Calculate Average True Range"""
    high = df['High']
    low = df['Low']
    close = df['Close']
   
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
   
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
   
    return atr

def create_atr_chart(df, period=14):
    """Create ATR chart"""
    atr = ATR(df, period)
   
    fig = go.Figure()
   
    # Add ATR line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=atr,
        name='ATR',
        line=dict(color='#2196F3', width=1)
    ))
   
    # Update layout with TradingView-like styling
    fig.update_layout(
        title=dict(
            text=f'<b>ATR ({period})</b>',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(color='#D9D9D9')
        ),
        xaxis=dict(
            title='Date',
            gridcolor='#2A2E39',
            zerolinecolor='#2A2E39',
            showgrid=True,
            rangeslider=dict(visible=False),
            type='date',
            tickfont=dict(color='#D9D9D9'),
            titlefont=dict(color='#D9D9D9')
        ),
        yaxis=dict(
            title='ATR',
            gridcolor='#2A2E39',
            zerolinecolor='#2A2E39',
            showgrid=True,
            tickfont=dict(color='#D9D9D9'),
            titlefont=dict(color='#D9D9D9')
        ),
        plot_bgcolor='#131722',
        paper_bgcolor='#131722',
        font=dict(color='#D9D9D9'),
        height=200,
        margin=dict(t=50, b=50, l=50, r=50),
        showlegend=False
    )
   
    return fig

app.layout = html.Div([
    # Main container with max-width and auto margins
    html.Div([
        # Header section
        html.Div([
            html.H1("Cryptocurrency Trading Signals Dashboard",
                   style={
                       "textAlign": "center",
                       "color": "#131722",
                       "fontSize": "clamp(1.5rem, 4vw, 2.5rem)",
                       "marginBottom": "1rem",
                       "textShadow": "1px 1px 2px rgba(0,0,0,0.1)"
                   }),
           
            # Controls section
            html.Div([
                # Cryptocurrency selector
                html.Div([
                    html.Label("Select Cryptocurrency",
                             style={"color": "#131722", "marginBottom": "0.5rem"}),
                    dcc.Dropdown(
                        id='crypto-dropdown',
                        options=stock_list,
                        value='BTC',
                        style={
                            "backgroundColor": "#FFFFFF",
                            "color": "#131722"
                        },
                        clearable=False
                    )
                ], style={"width": "100%", "marginBottom": "1rem"}),
               
                # Moving average controls
                html.Div([
                    # MA Type selector
                    html.Div([
                        html.Label("MA Type",
                                 style={"color": "#131722", "marginBottom": "0.5rem"}),
                        dcc.Dropdown(
                            id='ma-type-dropdown',
                            options=[{'label': ma, 'value': ma} for ma in ['SMA', 'EMA', 'WMA', 'VWMA']],
                            value='SMA',
                            style={
                                "backgroundColor": "#FFFFFF",
                                "color": "#131722"
                            },
                            clearable=False
                        )
                    ], style={"width": "100%", "marginBottom": "1rem"}),
                   
                    # Period selectors in a grid
                    html.Div([
                        # Short-term period
                        html.Div([
                            html.Label("Short-term Period",
                                     style={"color": "#131722", "marginBottom": "0.5rem"}),
                            dcc.Dropdown(
                                id='period1-dropdown',
                                options=[{'label': str(i), 'value': i} for i in [15, 20, 30]],
                                value=20,
                                style={
                                    "backgroundColor": "#FFFFFF",
                                    "color": "#131722"
                                },
                                clearable=False
                            )
                        ], style={"width": "100%", "marginBottom": "1rem"}),
                       
                        # Medium-term period
                        html.Div([
                            html.Label("Medium-term Period",
                                     style={"color": "#131722", "marginBottom": "0.5rem"}),
                            dcc.Dropdown(
                                id='period2-dropdown',
                                options=[{'label': str(i), 'value': i} for i in [50, 80, 100]],
                                value=50,
                                style={
                                    "backgroundColor": "#FFFFFF",
                                    "color": "#131722"
                                },
                                clearable=False
                            )
                        ], style={"width": "100%", "marginBottom": "1rem"}),
                       
                        # Long-term period
                        html.Div([
                            html.Label("Long-term Period",
                                     style={"color": "#131722", "marginBottom": "0.5rem"}),
                            dcc.Dropdown(
                                id='period3-dropdown',
                                options=[{'label': str(i), 'value': i} for i in [120, 150, 200]],
                                value=200,
                                style={
                                    "backgroundColor": "#FFFFFF",
                                    "color": "#131722"
                                },
                                clearable=False
                            )
                        ], style={"width": "100%", "marginBottom": "1rem"})
                    ], style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))",
                        "gap": "1rem",
                        "width": "100%"
                    })
                ], style={
                    "backgroundColor": "#F8F9FA",
                    "padding": "1.5rem",
                    "borderRadius": "8px",
                    "marginBottom": "2rem",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.05)",
                    "border": "1px solid #E9ECEF"
                }),
            ], style={"width": "100%", "maxWidth": "1200px", "margin": "0 auto"}),
           
            # Charts section
            html.Div([
                # Main price chart
                html.Div([
                    dcc.Graph(
                        id='trading-signals-chart',
                        style={
                            "width": "100%",
                            "height": "70vh",
                            "minHeight": "600px",
                            "margin": "0 auto"
                        },
                        config={
                            'responsive': True,
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                        }
                    )
                ], style={
                    "width": "100%",
                    "maxWidth": "1600px",
                    "marginBottom": "2rem",
                    "backgroundColor": "#FFFFFF",
                    "borderRadius": "8px",
                    "padding": "1rem",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.05)",
                    "border": "1px solid #E9ECEF",
                    "overflow": "hidden"
                }),
               
                # RSI chart
                html.Div([
                    dcc.Graph(
                        id='rsi-chart',
                        style={
                            "height": "25vh",
                            "width": "100%"
                        },
                        config={
                            'responsive': True,
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                        }
                    )
                ], style={
                    "width": "100%",
                    "marginBottom": "2rem",
                    "backgroundColor": "#FFFFFF",
                    "borderRadius": "8px",
                    "padding": "1rem",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.05)",
                    "border": "1px solid #E9ECEF"
                }),
               
                # MACD chart
                html.Div([
                    dcc.Graph(
                        id='macd-chart',
                        style={
                            "height": "25vh",
                            "width": "100%"
                        },
                        config={
                            'responsive': True,
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                        }
                    )
                ], style={
                    "width": "100%",
                    "marginBottom": "2rem",
                    "backgroundColor": "#FFFFFF",
                    "borderRadius": "8px",
                    "padding": "1rem",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.05)",
                    "border": "1px solid #E9ECEF"
                }),
               
                # ATR chart
                html.Div([
                    dcc.Graph(
                        id='atr-chart',
                        style={
                            "height": "25vh",
                            "width": "100%"
                        },
                        config={
                            'responsive': True,
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                        }
                    )
                ], style={
                    "width": "100%",
                    "marginBottom": "2rem",
                    "backgroundColor": "#FFFFFF",
                    "borderRadius": "8px",
                    "padding": "1rem",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.05)",
                    "border": "1px solid #E9ECEF"
                })
            ], style={"width": "100%", "maxWidth": "1200px", "margin": "0 auto"})
        ], style={"width": "100%", "padding": "1rem"})
    ], style={
        "maxWidth": "1400px",
        "margin": "0 auto",
        "padding": "1rem",
        "minHeight": "100vh",
        "background": "linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%)",
        "backgroundAttachment": "fixed"
    })
], style={
    "minHeight": "100vh",
    "background": "linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%)",
    "backgroundAttachment": "fixed",
    "position": "relative"
})

# Add custom CSS for better dropdown styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Custom styles for dropdowns */
            .Select-control {
                background-color: #FFFFFF !important;
                border: 1px solid #E9ECEF !important;
            }
            .Select-menu-outer {
                background-color: #FFFFFF !important;
                border: 1px solid #E9ECEF !important;
            }
            .Select-option {
                background-color: #FFFFFF !important;
                color: #131722 !important;
            }
            .Select-option:hover {
                background-color: #F8F9FA !important;
            }
            .Select-value-label {
                color: #131722 !important;
            }
            .Select-placeholder {
                color: #6C757D !important;
            }
            /* Smooth scrolling */
            html {
                scroll-behavior: smooth;
            }
            /* Better mobile experience */
            @media (max-width: 768px) {
                .Select-control {
                    font-size: 16px !important; /* Prevents zoom on iOS */
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

@app.callback(
    Output('trading-signals-chart', 'figure'),
    [Input('crypto-dropdown', 'value'),
     Input('ma-type-dropdown', 'value'),
     Input('period1-dropdown', 'value'),
     Input('period2-dropdown', 'value'),
     Input('period3-dropdown', 'value')]
)
def update_main_chart(crypto_symbol, ma_type, period1, period2, period3):
    df = load_crypto_data(crypto_symbol)
    if df is None:
        return go.Figure()
    main_chart = buy_n_sell(df, crypto_symbol, period1, period2, period3, ma_type)
    return main_chart

    


if __name__=='__main__':
    app.run(debug=True)



