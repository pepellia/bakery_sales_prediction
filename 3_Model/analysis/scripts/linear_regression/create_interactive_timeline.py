import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

# Define paths
MODEL_NAME = 'linear_regression_v8'
MODEL_OUTPUT = os.path.join(MODEL_ROOT, 'output', 'linear_regression', MODEL_NAME)
ANALYSIS_OUTPUT = os.path.join(MODEL_ROOT, 'analysis', 'results', 'linear_regression', MODEL_NAME)
ANALYSIS_VIZ = os.path.join(MODEL_ROOT, 'analysis', 'visualizations', 'linear_regression', MODEL_NAME)

def create_interactive_timeline():
    # Load the timeline data
    with open(os.path.join(ANALYSIS_OUTPUT, 'timeline_data.json'), 'r') as f:
        timeline_data = json.load(f)
    
    # Convert to pandas DataFrames
    train_daily = pd.DataFrame(timeline_data['training_period'])
    val_daily = pd.DataFrame(timeline_data['validation_period'])
    test_daily = pd.DataFrame(timeline_data['test_period'])
    
    # Convert dates
    train_daily['Datum'] = pd.to_datetime(train_daily['Datum'])
    val_daily['Datum'] = pd.to_datetime(val_daily['Datum'])
    test_daily['Datum'] = pd.to_datetime(test_daily['Datum'])
    
    # Create figure with subplots
    fig = make_subplots(rows=3, cols=1,
                       subplot_titles=('Training Period: Total Daily Sales (2013-08-01 to 2017-07-31)',
                                     'Validation Period: Total Daily Sales (2017-08-01 to 2018-07-31)',
                                     'Test Period: Predicted Total Daily Sales'))
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=train_daily['Datum'], y=train_daily['Umsatz'],
                  mode='lines', name='Training', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=val_daily['Datum'], y=val_daily['Umsatz'],
                  mode='lines', name='Validation', line=dict(color='green')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=test_daily['Datum'], y=test_daily['Umsatz'],
                  mode='lines', name='Test', line=dict(color='red')),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        title_text="Interactive Timeline of Sales",
        hovermode='x unified'
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Total Sales (€)")
    
    # Update x-axes labels
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    # Generate HTML file
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive Sales Timeline</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ margin: 0; padding: 20px; }}
            #timeline {{ width: 100%; height: 900px; }}
        </style>
    </head>
    <body>
        <div id="timeline">
            {fig.to_html(full_html=False, include_plotlyjs=False)}
        </div>
    </body>
    </html>
    """
    
    # Save the HTML file
    with open(os.path.join(ANALYSIS_VIZ, 'interactive_timeline.html'), 'w') as f:
        f.write(html_content)

if __name__ == "__main__":
    create_interactive_timeline()
