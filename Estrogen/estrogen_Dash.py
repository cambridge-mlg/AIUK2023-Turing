# conda env list
# conda create -n UK-AI python=3.8
# conda activate UK-AI
# conda list -n UK-AI or conda list
# conda install -n UK-AI pip
# conda install -c conda-forge dash
# conda install pymc3 Numpy=1.21.0 pickle pandas dash plotly scipy julia base64

import os
os.chdir('/home/yongchao/Desktop/UK-AI')

t = [0,1,2,3,4,6,8,10,12,16,24]; y = [0,25,100,132,90,82,60,55,32,15,4]

from julia.api import Julia
# from julia import Main

##### build the app ######
# lsof -i :8051
# kill -9 164233
# # check if the port is occupied
# import os
# import subprocess
# result = subprocess.run(['lsof', '-i', ':8051'], stdout=subprocess.PIPE)
# if result.returncode == 0:
#     # if the port is occupied, get the list of PIDs and kill them
#     output = result.stdout.decode('utf-8')
#     for line in output.split('\n'):
#         if 'LISTEN' in line:
#             pid = line.split()[1]
#             os.system(f'kill -9 {pid}')  

import numpy as np
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from scipy.integrate import solve_ivp
from julia.api import Julia
import base64

j = Julia(compiled_modules=False)
j.eval("""using MCMCChains; chain = read("/home/yongchao/Desktop/new julia/UK-AI/estrogen_chain.jls", Chains)""")

# load the figures
with open('/home/yongchao/Desktop/new julia/UK-AI/Estrogen/local_cloud.png', "rb") as local_cloud_file:
    local_cloud_data = base64.b64encode(local_cloud_file.read())
    local_cloud_data = local_cloud_data.decode()
    local_cloud_data = "{}{}".format("data:image/jpg;base64, ", local_cloud_data)
with open('/home/yongchao/Desktop/new julia/UK-AI/Estrogen/estrogen_equation.png', "rb") as estrogen_model_equation_file:
    estrogen_model_equation_data = base64.b64encode(estrogen_model_equation_file.read())
    estrogen_model_equation_data = estrogen_model_equation_data.decode()
    estrogen_model_equation_data = "{}{}".format("data:image/jpg;base64, ", estrogen_model_equation_data)
with open('/home/yongchao/Desktop/new julia/UK-AI/Estrogen/probabilistic_model.png', "rb") as probabilistic_model_file:
    probabilistic_model_data = base64.b64encode(probabilistic_model_file.read())
    probabilistic_model_data = probabilistic_model_data.decode()
    probabilistic_model_data = "{}{}".format("data:image/jpg;base64, ", probabilistic_model_data)
with open('/home/yongchao/Desktop/new julia/UK-AI/Estrogen/least_squares_fit.png', "rb") as least_squares_fit_file:
    least_squares_fit_data = base64.b64encode(least_squares_fit_file.read())
    least_squares_fit_data = least_squares_fit_data.decode()
    least_squares_fit_data = "{}{}".format("data:image/jpg;base64, ", least_squares_fit_data)
with open('/home/yongchao/Desktop/new julia/UK-AI/Estrogen/advantage.png', "rb") as advantage_file:
    advantage_data = base64.b64encode(advantage_file.read())
    advantage_data = advantage_data.decode()
    advantage_data = "{}{}".format("data:image/jpg;base64, ", advantage_data)

with open('/home/yongchao/Desktop/new julia/UK-AI/Estrogen/posterior_predictive_allData.png', "rb") as posterior_predictive_allData_file:
    posterior_predictive_all_data = base64.b64encode(posterior_predictive_allData_file.read())
    posterior_predictive_all_data = posterior_predictive_all_data.decode()
    posterior_predictive_all_data = "{}{}".format("data:image/jpg;base64, ", posterior_predictive_all_data)
with open('/home/yongchao/Desktop/new julia/UK-AI/Estrogen/posterior_predictive_singleData.png', 'rb') as posterior_predictive_singleData:
    posterior_predictive1_data = base64.b64encode(posterior_predictive_singleData.read())
    posterior_predictive1_data = posterior_predictive1_data.decode()
    posterior_predictive1_data = "{}{}".format("data:image/jpg;base64, ", posterior_predictive1_data)
with open('/home/yongchao/Desktop/new julia/UK-AI/Estrogen/posterior_predictive_twoData.png', 'rb') as posterior_predictive_twoData:
    posterior_predictive2_data = base64.b64encode(posterior_predictive_twoData.read())
    posterior_predictive2_data = posterior_predictive2_data.decode()
    posterior_predictive2_data = "{}{}".format("data:image/jpg;base64, ", posterior_predictive2_data)
with open('/home/yongchao/Desktop/new julia/UK-AI/Estrogen/posterior_predictive_threeData1.png', 'rb') as posterior_predictive31_file:
    posterior_predictive31_data = base64.b64encode(posterior_predictive31_file.read())
    posterior_predictive31_data = posterior_predictive31_data.decode()
    posterior_predictive31_data = "{}{}".format("data:image/jpg;base64, ", posterior_predictive31_data)
with open('/home/yongchao/Desktop/new julia/UK-AI/Estrogen/posterior_predictive_threeData2.png', 'rb') as posterior_predictive32_file:
    posterior_predictive32_data = base64.b64encode(posterior_predictive32_file.read())
    posterior_predictive32_data = posterior_predictive32_data.decode()
    posterior_predictive32_data = "{}{}".format("data:image/jpg;base64, ", posterior_predictive32_data)
# with open('/home/yongchao/Desktop/new julia/UK-AI/Estrogen/posterior_predictive_singleData.svg', 'rb') as posterior_predictive_singleData:
#     posterior_predictive1_data = base64.b64encode(posterior_predictive_singleData.read())
#     posterior_predictive1_data = posterior_predictive1_data.decode()
#     posterior_predictive1_data = "data:image/svg+xml;base64,{}".format(posterior_predictive1_data)
# with open('/home/yongchao/Desktop/new julia/UK-AI/Estrogen/posterior_predictive_twoData.svg', 'rb') as posterior_predictive_twoData:
#     posterior_predictive2_data = base64.b64encode(posterior_predictive_twoData.read())
#     posterior_predictive2_data = posterior_predictive2_data.decode()
#     posterior_predictive2_data = "data:image/svg+xml;base64,{}".format(posterior_predictive2_data)
# with open('/home/yongchao/Desktop/new julia/UK-AI/Estrogen/posterior_predictive_threeData1.svg', 'rb') as posterior_predictive31_file:
#     posterior_predictive31_data = base64.b64encode(posterior_predictive31_file.read())
#     posterior_predictive31_data = posterior_predictive31_data.decode()
#     posterior_predictive31_data = "data:image/svg+xml;base64,{}".format(posterior_predictive31_data)
# with open('/home/yongchao/Desktop/new julia/UK-AI/Estrogen/posterior_predictive_threeData2.svg', 'rb') as posterior_predictive32_file:
#     posterior_predictive32_data = base64.b64encode(posterior_predictive32_file.read())
#     posterior_predictive32_data = posterior_predictive32_data.decode()
#     posterior_predictive32_data = "data:image/svg+xml;base64,{}".format(posterior_predictive32_data)


# define the function that returns the data for the selected parameter
def get_parameter_data(parameter):
    global j
    return np.array(j.eval(f"chain[:{parameter}]")).flatten()
chain_dict = {}
for key in ["c_max","halflife","t_max"]:
    chain_dict[key] = get_parameter_data(key)


# Define the layout of the app
app = dash.Dash(__name__)

# Create a dropdown to select the parameter to plot
parameter_options = [{"label": "c_max", "value": "c_max"},
                    {"label": "halflife", "value": "halflife"},
                    {"label": "t_max", "value": "t_max"}]

page_1 = html.Div(style={'display': 'flex', 'flex-wrap': 'wrap'}, children=[
    # First row
    html.Div(style={'width': '100%', 'text-align': 'center'}, children=[
                        html.H1(children=['Estrogen Modelling using ', html.I('Turing.jl'), ' and Cloud Analytics Platform ', html.I('withdata')], style={'vertical-align': 'top', 'color': 'red'})
                    ]),

    # Left column
    html.Div(style={'width': '33%', 'text-align': 'left'}, children=[
        html.H2(['Advantages'], style={'text-align': 'center'}),
        html.Img(id="advantage", src=advantage_data, alt="my image", width="600", height="300", className="img_class"),
    ]),
    
    # Middle column
   html.Div(style={'width': '33%', 'text-align': 'center'}, children=[
        html.Label('c_max', style={'text-align': 'right'}),
        html.Div(dcc.Slider(id='c_max-slider', min=0, max=150, step=10, value=133)),
        html.Label('halflife', style={'text-align': 'right'}),
        html.Div(dcc.Slider(id='halflife-slider', min=0, max=10, step=1, value=3.5)),
        html.Label('t_max', style={'text-align': 'right'}),
        html.Div(dcc.Slider(id='t_max-slider', min=0, max=10, step=1, value=3), style={'padding-bottom': '20px'}),
        html.Img(id="estrogen_model_equation", src=estrogen_model_equation_data, alt="my image", width="400", height="120", className="img_class"),
    ]),
    
    # Right column
    html.Div(style={'width': '33%', 'text-align': 'right'}, children=[
        dcc.Graph(id='graph', figure={
            'data': [
                {'x': [], 'y': [], 'type': 'line', 'name': 'estrogen'},
                {'x': [], 'y': [], 'mode': 'markers', 'marker': {'symbol': 'cross', 'size': 10}, 'name': 'noisy measurements'},
            ],
            'layout': {
                'title': 'Estrogen Model Simulaiton',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Estrogen'},
            }
        })
    ]),

    ## the probabilistic model
    html.Div(style={'width': '50%', 'text-align': 'center'}, children=[        
                        html.H2(['Probabilistic Modelling using ', html.I('Turing.jl', style={'color': 'red'})], style={'text-align': 'center'}),
                        html.Img(id="probabilistic_model", src=probabilistic_model_data, alt="my image", width="600", height="500", className="img_class"),
                        ]),
    ## least squares fit
    html.Div(style={'width': '50%', 'text-align': 'center'}, children=[ 
                        html.H2(['Comparison: least squares fitting'], style={'text-align': 'center'}),
                        html.Img(id="probabilistic_model", src=least_squares_fit_data, alt="my image", width="600", height="300", className="img_class"),
                        ]),
])
# Define the callback for updating the plot in Page 1 when the slider values change
@app.callback(
    Output('graph', 'figure'),
    [Input('c_max-slider', 'value'),
     Input('halflife-slider', 'value'),
     Input('t_max-slider', 'value')])
def update_graph(c_max, halflife, t_max):
    
    def single_dose(c_max, halflife, t_max):
        f_vec = []
        global t
        for tt in t:
            if tt < t_max:
                f_vec.append(c_max/t_max * tt)
            else:
                f_vec.append(c_max * 2**(-(tt-t_max)/halflife))
        return f_vec
    sol = single_dose(c_max, halflife, t_max)
    return {
        'data': [
            {'x': t, 'y': sol, 'type': 'line', 'name': 'MODEL'},
            # {'x': t, 'y': sol+0.5*np.random.randn(len(sol)), 'mode': 'markers', 'marker': {'symbol': 'cross', 'size': 10}, 'name': 'noisy measurements'},
            {'x': t, 'y': y, 'mode': 'markers', 'marker': {'symbol': 'cross', 'size': 10}, 'name': 'noisy measurements'},
        ],
        'layout': {
            'title': 'The Estrogen Model',
            'xaxis': {'title': 'Time'},
            'yaxis': {'title': 'Estrogen'},
        }
    }

page_2 = html.Div(style={'display': 'flex', 'flex-wrap': 'wrap'}, children=[
    ## inference results
    # traces
    html.H2([html.I('withdata', style={'color': 'red'}), ' cloud-based data analytics platform: checkpoint-based continuous sampling'], style={'width': '100%','text-align': 'center'}),
    html.H3('From Cloud to Local: returning inference results', style={'width': '100%','text-align': 'center'}),
    html.Div(style={'width': '100%', 'display': 'flex'}, children=[
        html.Div(style={'width': '33.33%', 'text-align': 'right'}, children=[
            dcc.Graph(id='hist-c_max', figure=px.histogram(chain_dict['c_max'], nbins=50, histnorm="probability density", marginal="box", title="c_max Posterior Histogram"))
        ]),
        html.Div(style={'width': '33.33%', 'text-align': 'right'}, children=[
            dcc.Graph(id='hist-halflife', figure=px.histogram(chain_dict['halflife'], nbins=50, histnorm="probability density", marginal="box", title="halflife Posterior Histogram"))
        ]),
        html.Div(style={'width': '33.33%', 'text-align': 'right'}, children=[
            dcc.Graph(id='hist-t_max', figure=px.histogram(chain_dict['t_max'], nbins=50, histnorm="probability density", marginal="box", title="t_max Posterior Histogram"))
        ]),
    ]),
    # histograms
    html.Div(style={'width': '100%', 'display': 'flex'}, children=[
        html.Div(style={'width': '33.33%', 'text-align': 'right'}, children=[
            dcc.Graph(id='trace-c_max', figure=px.line(pd.DataFrame(chain_dict['c_max']), y=0, title="c_max Trajectory"))
        ]),
        html.Div(style={'width': '33.33%', 'text-align': 'right'}, children=[
            dcc.Graph(id='trace-halflife', figure=px.line(pd.DataFrame(chain_dict['halflife']), y=0, title="c_max Trajectory"))
        ]),
        html.Div(style={'width': '33.33%', 'text-align': 'right'}, children=[
            dcc.Graph(id='trace-t_max', figure=px.line(pd.DataFrame(chain_dict['t_max']), y=0, title="c_max Trajectory"))
        ]),
    ]),

    # posterior predictives
    html.Div(style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'}, children=[        
        # column 1    
        html.Div(style={'width': '100%', 'text-align': 'center'}, children=[        
        html.H2('Model Details', style={'text-align': 'center'}),
        html.Img(id="posterior predictive: all data", src=posterior_predictive_all_data, alt="my image", width="1200", height="750", className="img_class"),
        ]),
    ]),

    # data uncertainty
    html.Div(style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'}, children=[        
        # column 1    
        html.Div(style={'width': '50%', 'text-align': 'center'}, children=[        
        html.H2('single data point', style={'text-align': 'center'}),
        html.Img(id="posterior predictive: single", src=posterior_predictive1_data, alt="my image", width="400", height="200", className="img_class"),
        ]),
        # column 2
        html.Div(style={'width': '50%', 'text-align': 'center'}, children=[
        html.H2('two data points', style={'text-align': 'center'}),
        html.Img(id="posterior predictive: two data points", src=posterior_predictive2_data, alt="my image", width="400", height="200", className="img_class"),
        ]),
        # column 1    
        html.Div(style={'width': '50%', 'text-align': 'center'}, children=[        
        html.H2('Three data points', style={'text-align': 'center'}),
        html.Img(id="posterior predictive: three wo data points 1", src=posterior_predictive31_data, alt="my image", width="400", height="200", className="img_class"),
        ]),
        # column 2
        html.Div(style={'width': '50%', 'text-align': 'center'}, children=[
        html.H2('Three data points', style={'text-align': 'center'}),
        html.Img(id="posterior predictive: three wo data points 2", src=posterior_predictive32_data, alt="my image", width="400", height="200", className="img_class"),
        ]),
    ]),
])



# Define the app layout
page_dropdown_options = [
    {'label': 'Page 1', 'value': 'page-1'},
    {'label': 'Page 2', 'value': 'page-2'}
]

# Define the app layout
app.layout = html.Div([
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

if __name__ == '__main__':
    app.run_server('0.0.0.0', port=8051, debug=True)