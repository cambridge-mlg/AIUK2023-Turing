# import Python libraries
import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, recall_score, accuracy_score, precision_score, f1_score

# helper functions
def classification_evaluate(y_test, y_pred_probs, threshold=0.5):
    """
    Evaluates a binary classification model by applying a threshold to predicted probabilities
    and calculating the confusion matrix and several performance metrics.

    Parameters:
    y_test (array-like): True binary labels for the test data
    y_pred_probs (array-like): Predicted probabilities for the test data
    threshold (float): Threshold value between 0 and 1 used to convert probabilities to binary predictions.
                        Default is 0.5.

    Returns:
    tuple: A tuple containing the confusion matrix DataFrame and the summary metrics DataFrame.
    """
    # Apply threshold to predicted probabilities
    y_pred = np.where(y_pred_probs >= threshold, 1, 0)

    # Calculate the confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)
    df_confusion_mat = pd.DataFrame(confusion_mat, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])

    # Calculate accuracy, sensitivity (recall), precision, and F1 score
    accuracy = (confusion_mat[0][0] + confusion_mat[1][1]) / sum(sum(confusion_mat))
    sensitivity = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Create a pandas DataFrame to summarize the metrics
    df_metrics = pd.DataFrame({'Accuracy': accuracy, 'Sensitivity (Recall)': sensitivity,
                               'Precision': precision, 'F1 Score': f1}, index=[0])

    return df_confusion_mat, df_metrics
def plot_roc_curve_matplot(y_true, y_pred_probs):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
def plot_roc_curve_plotly(y_true, y_pred_probs):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name='ROC curve (area = %0.2f)' % roc_auc, mode='lines', line=dict(color='darkorange', width=2)), row=1, col=1)
    fig.add_shape(type='line', x0=0, x1=1, y0=0, y1=1, yref='y', xref='x', line=dict(color='navy', width=2, dash='dash'))
    fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', title='Receiver Operating Characteristic (ROC) Curve', showlegend=True, legend=dict(x=0.7, y=0.1))
    return fig

# set local working directory
# wd = os.getcwd() + "/Insurance" # localPath = "/home/yongchao/Desktop/AIUK"
wd = "/home/yongchao/Desktop/AIUK" + "/Insurance" 
results_folder = wd + "/results"
images_folder = wd + "/images"
# point to Github repo for retrieving data
githubRepo = "https://github.com/YongchaoHuang/AIUK"

# setup for Julia
from julia.api import Julia
j = Julia(compiled_modules=False)

# load JUlia pre-processed data
X_df_train = pd.read_json(results_folder+"/X_df_train.json", orient='records')
X_df_test = pd.read_json(results_folder+"/X_df_test.json", orient='records')
y_train = pd.read_json(results_folder+"/y_train.json", orient='records').to_numpy().flatten()
y_test = pd.read_json(results_folder+"/y_test.json", orient='records').to_numpy().flatten()

df = pd.read_json(results_folder+"/df.json", orient='records')
numeric_cols = pd.read_json(results_folder+"/numeric_cols.json", orient='records').to_numpy().flatten()
df_fraud = df.loc[df.loc[:, "fraud_reported"]=="Y", :]
df_non_fraud = df.loc[df.loc[:, "fraud_reported"]=="N", :]

# load Turing inference results, save it to a JSON file, load it into Python.
j.eval(f"""using MCMCChains, JSON, JLD2;
chain = read(joinpath("{results_folder}", "insurance_chain_turing.jls"), Chains);
keys(chain);
chain[\"intercept\"];
pairs(chain);

# convert MCMCChain object to a dictionary
chain_dict = Dict(Symbol(param) => chain[param][:] for param in keys(chain));

# save the dictionary to a JSON file
open(joinpath("{results_folder}", "chain_dict.json"), "w") do io
    write(io, JSON.json(chain_dict))
end
""")
# load the JSON trace file in Python as a Python dictionary
chain_dict_json_python_file_path = os.path.join(results_folder, "chain_dict.json")
with open(chain_dict_json_python_file_path, "r") as f:
    chain_dict = json.load(f)
print(chain_dict)
chain_dict.keys()

# (3) Load predictions
# j.eval(f"""using JLD2; y_test_preds_prob=read(joinpath("{results_folder}","y_test_preds_prob.jld2"))""")
j.eval(f"""@load joinpath("{results_folder}","y_test_preds_prob.jld2") y_test_preds_prob""")
y_test_preds_prob_julia_array = j.eval("y_test_preds_prob")
# py_array = julia_array.tolist()
y_test_preds_prob = np.array(y_test_preds_prob_julia_array)
y_test_preds_prob.shape

# Plot the ROC curve
plot_roc_curve_matplot(y_test, y_test_preds_prob)

# confusion matrix and summary metrics
confusion_mat, metrics = classification_evaluate(y_test, y_test_preds_prob, threshold=0.6)
print('Confusion Matrix:\n', confusion_mat, '\n')
print('Summary Metrics:\n', metrics)

# Number of claims dropdown
observe_variables = ["insured_sex",
                     "insured_education_level",
                     "insured_occupation",
                     "insured_relationship",
                     "incident_type",
                     "collision_type",
                     "incident_severity",
                     "incident_state",
                     "property_damage",
                     "police_report_available",
                     "witnesses",
                     "auto_make"]


##### build the app ######  
import numpy as np
import pandas as pd
import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from julia.api import Julia
import base64

## Results from the data pre-processing pipeline ####

# load static figures from ./images
with open(images_folder+'/gender_vs_total_claim_amount.png', "rb") as file:
    file_data = base64.b64encode(file.read())
    file_data = file_data.decode()
    png1_data = "{}{}".format("data:image/jpg;base64, ", file_data)
with open(images_folder+'/variable pairplot.png', "rb") as file:
    file_data = base64.b64encode(file.read())
    file_data = file_data.decode()
    png2_data = "{}{}".format("data:image/jpg;base64, ", file_data)
with open(images_folder+'/no of claims by state.png', "rb") as file:
    file_data = base64.b64encode(file.read())
    file_data = file_data.decode()
    png3_data = "{}{}".format("data:image/jpg;base64, ", file_data)
with open(images_folder+'/Education level vs total claim amount.svg', "rb") as file:
    file_data = base64.b64encode(file.read())
    file_data = file_data.decode()
    svg1_data = "data:image/svg+xml;base64,{}".format(file_data)
with open(images_folder+'/numeric corrplot.svg', "rb") as file:
    file_data = base64.b64encode(file.read())
    file_data = file_data.decode()
    svg2_data = "data:image/svg+xml;base64,{}".format(file_data)


with open(images_folder+'/model.svg', "rb") as file:
    file_data = base64.b64encode(file.read())
    file_data = file_data.decode()
    svg3_data = "data:image/svg+xml;base64,{}".format(file_data)
with open(images_folder+'/Test performance: predicted labels.svg', "rb") as file:
    file_data = base64.b64encode(file.read())
    file_data = file_data.decode()
    svg4_data = "data:image/svg+xml;base64,{}".format(file_data)
with open(images_folder+'/Test performance: predicted probability.svg', "rb") as file:
    file_data = base64.b64encode(file.read())
    file_data = file_data.decode()
    svg5_data = "data:image/svg+xml;base64,{}".format(file_data)

with open(images_folder+'/claims data.svg', "rb") as file:
    file_data = base64.b64encode(file.read())
    file_data = file_data.decode()
    svg9_data = "data:image/svg+xml;base64,{}".format(file_data)

# Define the layout of the app
app = dash.Dash(__name__)
pio.templates.default = "plotly_white" #plotly_dark, ggplot2, seaborn, simple_white, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, none

page_1 = html.Div(style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-evenly', 'align-items': 'center'}, children=[
                    html.Div(style={'height': '200px'}), # space div: doing nothing
                    # First row
                    html.Div(style={'width': '100%', 'text-align': 'center'}, children=[html.H1(children=['Insurance fraud detection using the probabilistic programming language ',  html.Span('Turing.jl', style={'color': 'red', 'font-style': 'italic'}), ' and Cloud Analytics Platform ', html.Span('withdata', style={'color': 'red', 'font-style': 'italic'})], style={'vertical-align': 'top', 'color': 'purple'})]),

                    # new row
                    html.H2(children=['Data Pre-processing Pipeline: Cleaning, Missing Imputation, Exploratory Data Analysis, Feature Engineering ...'], style={'width': '100%', 'text-align': 'center', 'color': 'black'}),
                    
                    # new row
                    html.Div(style={'width': '70%','display': 'flex','justify-content': 'center'}, children=[
                        html.Img(id="svg9", src=svg9_data, alt="my image", width="1000", height="700", className="img_class"),
                    ]),
                    html.Div(style={'width': '30%'}, children=[
                        dcc.Graph(
                            id='fraud-pie-chart',
                            figure=go.Figure(
                                go.Pie(
                                    labels=['Non-Fraud', 'Fraud'],
                                    values=[df['fraud_reported'].value_counts()[0], df['fraud_reported'].value_counts()[1]],
                                    hole=0.5,
                                    textinfo='label+percent+value',
                                    marker=dict(colors=['tab:green', 'tab:red']),
                                ),
                            ).update_layout(
                                title='Count of Fraud and Non-Fraud Cases',
                                height=400,  # set the height of the plot to 400 pixels
                                width=500,  # set the width of the plot to 500 pixels
                                title_font=dict(size=20)  # set the font size of title to 20
                            )
                        )
                    ]),

                    # new row
                    html.H2("total_claim_amount", style={'width': '100%', 'text-align': 'center', 'font-style': 'italic'}), 

                    # new row
                    html.Div(style={'width': '33%'}, children=[html.Img(id="png1", src=png1_data, alt="my image", width="500", height="300", className="img_class", style={'text-align': 'right'})]),
                    html.Div(style={'width': '33%'}, children=[dcc.Graph(
                            id='box-plot-1',
                            figure=go.Figure(
                                go.Box(
                                    x=df['insured_sex'],
                                    y=df['total_claim_amount'],
                                    name='Insured Sex',
                                ),
                            ),
                        ),
                    ]),
                    html.Div(style={'width': '33%'}, children=[html.Img(id="svg1", src=svg1_data, alt="my image", width="600", height="400", className="img_class", style={'text-align': 'right'})]),

                    # new row 
                    html.H2("Number of fraud & total claim amount breakdown", style={'width': '100%', 'text-align': 'center', 'font-style': 'italic'}), 
                    # claim number dropdown
                    html.Div(style={'width': '50%'}, children=[
                        dcc.Dropdown(
                            id='no_fraud_dropdown',
                            options=[{'label': var, 'value': var} for var in observe_variables],
                            value=observe_variables[0]
                        ),
                        dcc.Graph(id='no-fraud-bar-chart')
                    ]),

                    html.Div(style={'width': '50%'}, children=[
                        dcc.Dropdown(
                            id='claim_amount_dropdown',
                            options=[{'label': var, 'value': var} for var in observe_variables],
                            value=observe_variables[0]
                        ),
                        dcc.Graph(id='claim-amount-violin-plot')
                    ]),

                    html.Div(style={'height': '100px'}), # space div: doing nothing
                    
                    # new row
                    html.Div(style={'width': '40%'}, children=[html.H3("Feature correlation", style={'text-align': 'center', 'font-style': 'italic'}),
                                                               html.Img(id="png2", src=png2_data, alt="my image", width="600", height="400", className="img_class", style={'text-align': 'center'})]),
                    html.Div(style={'width': '40%'}, children=[
                        dcc.Dropdown(
                            id='density-dropdown',
                            options=[{'label': col, 'value': col} for col in df.select_dtypes(include=['float', 'int']).columns],
                            value='age'
                        ),
                        dcc.Graph(
                            id='density-plot'
                        ),
                    ]),
                ])

page_2 = html.Div(style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-evenly', 'align-items': 'center'}, children=[
                    ## inference results: no callbacks
                    html.H1([html.I('withdata', style={'color': 'red'}), ' cloud-based data analytics platform: checkpoint-based continuous sampling'], style={'width': '100%','text-align': 'center'}),
                    
                    ## the probabilistic model
                    html.Div(style={'width': '100%', 'text-align': 'center'}, children=[        
                        html.H2(['Probabilistic Modelling using ', html.I('Turing.jl', style={'color': 'red'})], style={'text-align': 'center'}),
                        html.Img(id="probabilistic_model", src=svg3_data, alt="my image", width="900", height="700", className="img_class"),
                        ]), 
                    
                    # histograms
                    html.H2('From Cloud to Local: returning inference results', style={'width': '100%','text-align': 'center'}),
                    html.Div(style={'width': '30%','text-align': 'center'}, children=[
                        dcc.Dropdown(
                            id='coefficients-dropdown',
                            options=[{'label': f'coefficients[{i}]', 'value': f'coefficients[{i}]'} for i in range(1, 21)],
                            value='coefficients[1]'
                        ),
                    ]),
                    html.Div(style={'width': '100%'}, children=[
                        html.Div(style={'display': 'flex'}, children=[
                            html.Div(style={'width': '50%'}, id='hist-coefficients'),
                            html.Div(style={'width': '50%'}, id='trace-coefficients')
                        ]),                    
                    ])
                ])

page_3 = html.Div(style={'display': 'flex', 'flex-wrap': 'wrap', 'width': '100%', 'justify-content': 'center'}, children=[        
                    html.H1('Test Performance & Informed Decision Making', style={'vertical-align': 'top', 'color': 'purple'}),
                    
                    html.Div(style={'width': '100%', 'justify-content': 'center'}, children=[html.H2('Skillfulness of classifier')]),
                    html.Div(style={'width': '100%', 'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center'}, children=[
                        html.Div(style={'width': '30%', 'justify-content': 'center'}, children=[
                            dcc.Graph(figure=plot_roc_curve_plotly(y_test, y_test_preds_prob), id='roc-curve')
                        ]),
                        html.Div(style={'width': '30%'}, children=[
                            html.H3("Confusion Matrix, Recall, Accuracy, Precision"),
                            html.Div(id='output-metrics')
                        ]),
                        html.Div(style={'width': '40%'}, children=[
                            dcc.Graph(
                                id='scatter-plot',
                                figure=go.Figure(
                                    go.Scatter(
                                        x=np.arange(len(y_test_preds_prob)),
                                        y=y_test_preds_prob,
                                        mode='markers',
                                        marker=dict(
                                            color=np.where(y_test_preds_prob > 0.5, 'red', 'black'),
                                            symbol=np.where(y_test == 1, 'star', 'circle'),
                                            size=np.where(y_test == 1, 10, 5),
                                        )
                                    ),
                                    layout=go.Layout(
                                        title='Predicted Probabilities',
                                        xaxis=dict(title='Customer ID'),
                                        yaxis=dict(title='Probability'),
                                        shapes=[
                                            # Add horizontal line at y=0
                                            dict(
                                                type='line',
                                                x0=0,
                                                x1=len(y_test_preds_prob),
                                                y0=0,
                                                y1=0,
                                                line=dict(color='gray', dash='dash')
                                            ),
                                            # Add horizontal line at y=1
                                            dict(
                                                type='line',
                                                x0=0,
                                                x1=len(y_test_preds_prob),
                                                y0=1,
                                                y1=1,
                                                line=dict(color='gray', dash='dash')
                                            )
                                        ]
                                    )
                                ),
                            ),
                        ]),
                    ]),
                    
                    html.Div(style={'width': '100%', 'justify-content': 'center'}, children=[html.H2("User-defined threshold to suit different business needs", style={'width': '100%','text-align': 'center'})]),
                    
                    html.Div(style={'width': '25%', 'justify-content': 'center'}, children=[
                        dcc.Slider(
                            id='threshold-slider',
                            min=0,
                            max=1,
                            step=0.01,
                            value=0.5,
                            marks={0: '0', 0.5: '0.5', 1: '1'}
                        ),
                    ]),
                ])

page_dropdown_options = [
    {'label': 'Page 1', 'value': 'page-1'},
    {'label': 'Page 2', 'value': 'page-2'},
    {'label': 'Page 3', 'value': 'page-3'},
]

# Define the app layout
# 'backgroundColor': 'white', 'black', 'rgb(230, 230, 230)', 'rgb(230, 230, 230)'
app.layout = html.Div(style={'backgroundColor': 'white'}, children=[
    # Create a placeholder div for the page content
    html.Div(id='page-content'),
    
    # Create the dropdown menu on the top right corner
    html.Div([
        dcc.Dropdown(
            id='page-dropdown',
            options=page_dropdown_options,
            value='page-1',
            clearable=False,
            style={'width': '150px'}
        )
    ], style={'position': 'fixed', 'top': 10, 'right': 10}),
])
# define the callback to update the page content based on the selected dropdown value
@app.callback(Output('page-content', 'children'), [Input('page-dropdown', 'value')])
def display_page(value):
    if value == 'page-1':
        return page_1
    elif value == 'page-2':
        return page_2
    elif value == 'page-3':
        return page_3


# Define the callback to update the grouped bar chart when the dropdown value changes for observe_var in page_1
@app.callback(
    dash.dependencies.Output('no-fraud-bar-chart', 'figure'),
    [dash.dependencies.Input('no_fraud_dropdown', 'value')]
)
def update_grouped_bar_chart(observe_var):
    # Group the data by the selected variable and fraud_reported
    df_groupby = df.groupby([observe_var, 'fraud_reported']).size().reset_index(name='count')
    
    # Create the grouped bar chart
    trace1 = go.Bar(
        x=df_groupby[df_groupby['fraud_reported'] == 'N'][observe_var],
        y=df_groupby[df_groupby['fraud_reported'] == 'N']['count'],
        name='Non-Fraud',
        marker=dict(color='green')
    )
    trace2 = go.Bar(
        x=df_groupby[df_groupby['fraud_reported'] == 'Y'][observe_var],
        y=df_groupby[df_groupby['fraud_reported'] == 'Y']['count'],
        name='Fraud',
        marker=dict(color='red')
    )
    return {
        'data': [trace1, trace2],
        'layout': go.Layout(
            title=f"Number of claims by {observe_var}",
            xaxis=dict(title=observe_var),
            yaxis=dict(title='Number'),
            barmode='group'
        )
    }

# Define the callback to update the grouped violin plot when the dropdown value changes
@app.callback(
    dash.dependencies.Output('claim-amount-violin-plot', 'figure'),
    [dash.dependencies.Input('claim_amount_dropdown', 'value')]
)
def update_violin_plot(x_var):
    # Create a list of traces for each category of the selected variable
    traces = []
    for category in df[x_var].unique():
        df_category = df[df[x_var] == category]
        non_fraud_data = df_category[df_category['fraud_reported'] == 'N']['total_claim_amount']
        fraud_data = df_category[df_category['fraud_reported'] == 'Y']['total_claim_amount']
        
        trace1 = go.Violin(
            x=[category],
            y=non_fraud_data,
            name='',
            box=dict(visible=True),
            meanline=dict(visible=True),
            points='all',
            jitter=0.05,
            side='negative',
            scalemode='width',
            width=0.4,
            spanmode='hard',
            marker=dict(color='green')
        )
        
        trace2 = go.Violin(
            x=[category],
            y=fraud_data,
            name='',
            box=dict(visible=True),
            meanline=dict(visible=True),
            points='all',
            jitter=0.05,
            side='positive',
            scalemode='width',
            width=0.4,
            spanmode='hard',
            marker=dict(color='red')
        )
        
        traces.append(trace1)
        traces.append(trace2)

    # Set the layout and return the figure
    layout = go.Layout(
        title='Total Claim Amount by {}'.format(x_var),
        xaxis=dict(title=x_var),
        yaxis=dict(title='Total Claim Amount', zeroline=False),
        violinmode='group',
        showlegend=False
    )
    
    # Add custom legend
    legend_trace1 = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color='green'),
        showlegend=True,
        name='Non-Fraud'
    )
    
    legend_trace2 = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color='red'),
        showlegend=True,
        name='Fraud'
    )
    
    traces.append(legend_trace1)
    traces.append(legend_trace2)
    
    return {'data': traces, 'layout': layout}


# Define the callback to update the Divs when the dropdown value changes in page_2
@app.callback(
    [Output('hist-coefficients', 'children'),
     Output('trace-coefficients', 'children')],
    [Input('coefficients-dropdown', 'value')]
)
def update_divs(selected_value):
    hist_fig = px.histogram(chain_dict[selected_value], nbins=50, histnorm="probability density", marginal="box", title="Coefficient Posterior Histogram")
    hist_graph = dcc.Graph(id='hist-graph', figure=hist_fig)
    trace_fig = px.line(pd.DataFrame(chain_dict[selected_value]), y=0, title="Coefficients Trajectory")
    trace_graph = dcc.Graph(id='trace-graph', figure=trace_fig)
    return hist_graph, trace_graph

# Define the callback for updating the plot and metrics in page_3 when the slider values change
@app.callback(
    [Output('scatter-plot', 'figure'), Output('output-metrics', 'children')],
    [Input('threshold-slider', 'value')]
)
def update_scatter_plot(threshold):
    # Update the color of the markers based on the threshold
    marker_color = np.where(y_test_preds_prob > threshold, 'red', 'black')
    marker_symbol = np.where(y_test == 1, 'star', 'circle')

    # Update the layout to include the threshold line
    layout = go.Layout(
        title='Logistic Regression Predicted Probabilities',
        xaxis=dict(title='Customer ID'),
        yaxis=dict(title='Probability'),
        shapes=[
            # Add horizontal line at y=0
            dict(
                type='line',
                x0=0,
                x1=len(y_test_preds_prob),
                y0=0,
                y1=0,
                line=dict(color='gray', dash='dash')
            ),
            # Add horizontal line at y=1
            dict(
                type='line',
                x0=0,
                x1=len(y_test_preds_prob),
                y0=1,
                y1=1,
                line=dict(color='gray', dash='dash')
            ),
            # Add horizontal line at user-defined threshold
            dict(
                type='line',
                x0=0,
                x1=len(y_test_preds_prob),
                y0=threshold,
                y1=threshold,
                line=dict(color='red')
            )
        ]
    )
    
    # Create the updated figure with the new marker colors and layout
    fig = go.Figure(
        go.Scatter(
            x=np.arange(len(y_test_preds_prob)),
            y=y_test_preds_prob,
            mode='markers',
            marker=dict(
                color=marker_color,
                size=np.where(y_test == 1, 10, 5),
                symbol=marker_symbol
            )
        ),
        layout=layout
    )

    y_pred = np.where(y_test_preds_prob >= threshold, 1, 0)
    confusion_mat = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    # Return the updated figure and output metrics
    return fig, [
        html.P(f"Confusion Matrix: {confusion_mat}"),
        html.P(f"Recall: {recall:.2f}"),
        html.P(f"Accuracy: {accuracy:.2f}"),
        html.P(f"Precision: {precision:.2f}")
    ]

# Define the callback to update the density plot when the dropdown value changes
@app.callback(
    dash.dependencies.Output('density-plot', 'figure'),
    [dash.dependencies.Input('density-dropdown', 'value')]
)
def update_density_plot(column):
    # Create a kernel density plot for the selected column
    fig = px.histogram(df, x=column, color='fraud_reported', nbins=50, histfunc='count')
    # fig.update_layout(title=f'Histogram for {column}')
    return fig

if __name__ == '__main__':
    app.run_server('0.0.0.0', port=8051, debug=True)