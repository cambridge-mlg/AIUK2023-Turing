# set local working directory
import os
cwd = os.getcwd() # localPath = "/home/yongchao/Desktop/AIUK"
results_folder = cwd + "/results"
images_folder = cwd + "/images"
# point to Github repo for retrieving data
githubRepo = "https://github.com/YongchaoHuang/AIUK"

#### Load the saved inference results: PyMC3 or Turing ####
inference_engine = "PyMC3"

if inference_engine == "PyMC3":
    import pymc3 as pm
    import pickle
    import numpy as np
    with open(results_folder+"/battery_model_pymc.pkl", "rb") as f:
        pm_model = pickle.load(f)
        trace = pm.load_trace(directory=results_folder+"/battery_trace_pymc", model=pm_model)
if inference_engine == "Turing":
    os.chdir(results_folder)
    from julia.api import Julia
    j = Julia(compiled_modules=False)
    j.eval("""using JLS;
            chain = read("battery_chain_turing.jls")
        """)

#### the constitutive simulator: for slider use ####
class towRCs_ECM:
    def __init__(self, rt, r1_, t1, r2_, t2, omega):
        """
        Args:
            - Rt: the total resistance of the battery (aka R at f=0 [Hz])
            - rt: the normalised DC resistance
            - ri: the normalised AC resistance of the i-th RC pair
            - ti: the normalised time constant of the i-th RC pair
            - sigma: experimental noise variance
            - omega: angular frequencies [rad/s]
        """
        self.omega = omega
        self.omega_mu = np.mean(np.log(self.omega))
        self.omega_sigma = np.std(np.log(self.omega))
        
        self.rt = rt
        self.t1 = t1
        self.r1 = np.exp(-np.exp(r1_))
        self.t2 = t2
        self.r2 = np.exp(-np.exp(r2_))
        self.Rt = np.exp(self.rt)
        self.r0 = 1 - self.r1 - self.r2
   
    def real_part(self):
        """
        Returns: real part of impedance spectrum
        """
        return self.Rt * (
            self.r0 + self.r1 / 2 * (1 - np.tanh(np.log(self.omega) - (self.omega_sigma * self.t1 + self.omega_mu)))
            + self.r2 / 2 * (1 - np.tanh(np.log(self.omega) - (self.omega_sigma * self.t2 + self.omega_mu)))
        )

    def imaginary_part(self):
        """
        Returns: imaginary part of impedance spectrum
        """
        return self.Rt * (
            (self.r1 / 2) / np.cosh(np.log(self.omega) - (self.omega_sigma * self.t1 + self.omega_mu))
            + (self.r2 / 2) / np.cosh(np.log(self.omega) - (self.omega_sigma * self.t2 + self.omega_mu))
        )

# instantiate the battery model, and generate simulated ground truth data.
n = 100
f = np.logspace(1, 10, n)  # frequency [Hz]
angular_f = 2 * np.pi * f
rt, r1, t1, r2, t2, noise_std = [2, -0.5, -1, 0, 0.5, 0.01]
twoRCs_model_obj = towRCs_ECM(rt, r1, t1, r2, t2, omega=angular_f)

noise = np.random.normal(0, noise_std, n)
reals_simulated = twoRCs_model_obj.real_part() + noise
ims_simulated = twoRCs_model_obj.imaginary_part() +noise

log_angular_f = np.log10(twoRCs_model_obj.omega / (2 * np.pi))

#### lifespan estimation ####
def calculate_remaining_life(rt, r1, t1, r2, t2):
    global angular_f
    twoRCs_model_obj = towRCs_ECM(rt, r1, t1, r2, t2, omega=angular_f)
    # simulate the battery's behavior for different loads and temperatures
    load = 10  # load [W]
    temp = 25  # temperature [C]
    capacity = 100  # battery capacity [Ah]
    voltage = 12  # battery voltage [V]

    current = voltage / (twoRCs_model_obj.real_part() + 1j * twoRCs_model_obj.imaginary_part()) * load
    power = voltage * current

    # estimate the battery's lifespan based on the current and power output
    lifespan = (capacity / np.trapz(current, dx=1)).real*100  # assuming a constant current load
    return np.abs(lifespan)
r1_simulated_vec = np.linspace(0, 5, 100)
r1_remaining_life_vec = []
for r1_simulated in r1_simulated_vec:
    r1_remaining_life_vec.append(calculate_remaining_life(rt, r1_simulated, t1, r2, t2))

r2_simulated_vec = np.linspace(0, 5, 100)
r2_remaining_life_vec = []
for r2_simulated in r1_simulated_vec:
    r2_remaining_life_vec.append(calculate_remaining_life(rt, r1, t1, r2_simulated, t2))

t1_simulated_vec = np.linspace(0, 5, 100)
t1_remaining_life_vec = []
for t1_simulated in t1_simulated_vec:
    t1_remaining_life_vec.append(calculate_remaining_life(rt, r1, t1_simulated, r2, t2))

t2_simulated_vec = np.linspace(0, 5, 100)
t2_remaining_life_vec = []
for t2_simulated in t2_simulated_vec:
    t2_remaining_life_vec.append(calculate_remaining_life(rt, r1, t1, r2, t2_simulated))

rt_simulated_vec = np.linspace(-0.6, 5, 100)
rt_remaining_life_vec = []
for rt_simulated in rt_simulated_vec:
    rt_remaining_life_vec.append(calculate_remaining_life(rt_simulated, r1, t1, r2, t2))


##### build the app ######  

import numpy as np
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from julia.api import Julia
import base64

# load static figures from /images
with open(images_folder+'/battery 2RC diagram.svg', "rb") as PK_model_diagram_file:
    battery_model_diagram_data = base64.b64encode(PK_model_diagram_file.read())
    battery_model_diagram_data = battery_model_diagram_data.decode()
    battery_model_diagram_data = "data:image/svg+xml;base64,{}".format(battery_model_diagram_data)
with open(images_folder+'/battery model equation.svg', "rb") as battery_model_equation_file:
    battery_model_equation_data = base64.b64encode(battery_model_equation_file.read())
    battery_model_equation_data = battery_model_equation_data.decode()
    battery_model_equation_data = "data:image/svg+xml;base64,{}".format(battery_model_equation_data)
with open(images_folder+'/battery model details.png', "rb") as PK_model_details1_file:
    PK_model_details1_data = base64.b64encode(PK_model_details1_file.read())
    PK_model_details1_data = PK_model_details1_data.decode()
    PK_model_details1_data = "{}{}".format("data:image/jpg;base64, ", PK_model_details1_data)
with open(images_folder+'/battery probabilistic model.png', "rb") as probabilistic_model_file:
    probabilistic_model_data = base64.b64encode(probabilistic_model_file.read())
    probabilistic_model_data = probabilistic_model_data.decode()
    probabilistic_model_data = "{}{}".format("data:image/jpg;base64, ", probabilistic_model_data)

with open(images_folder+'/battery management system 1.svg', "rb") as BMS_file1:
    BMS_data1 = base64.b64encode(BMS_file1.read())
    BMS_data1 = BMS_data1.decode()
    BMS_data1 = "data:image/svg+xml;base64,{}".format(BMS_data1)
with open(images_folder+'/battery management system 2.svg', "rb") as BMS_file2:
    BMS_data2 = base64.b64encode(BMS_file2.read())
    BMS_data2 = BMS_data2.decode()
    BMS_data2 = "data:image/svg+xml;base64,{}".format(BMS_data2)
with open(images_folder+'/battery lifespan estimation.svg', "rb") as battery_lifespan_file:
    battery_lifespan_data = base64.b64encode(battery_lifespan_file.read())
    battery_lifespan_data = battery_lifespan_data.decode()
    battery_lifespan_data = "data:image/svg+xml;base64,{}".format(battery_lifespan_data)

# define the function that returns the data for the selected parameter
def get_parameter_data(parameter):
    return np.array(trace[parameter]).flatten()
chain_dict = {}
for key in ["rt","t1","r1","t2","r2","noise_std"]:
    chain_dict[key] = get_parameter_data(key)

true_value = {"rt":rt,"t1":t1,"r1":r1,"t2":t2,"r2":r2,"noise_std":noise_std}

# Define the layout of the app
app = dash.Dash(__name__)
pio.templates.default = "plotly_white" #plotly_dark, ggplot2, seaborn, simple_white, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, none

# Create a dropdown to select the parameter to plot
parameter_options = [{"label": "rt", "value": "rt"},
                        {"label": "r1", "value": "r1"},
                        {"label": "t1", "value": "t1"},
                        {"label": "r2", "value": "r2"},
                        {"label": "t2", "value": "t2"},
                        {"label": "noise_std", "value": "noise_std"}]

dropdown_options = [
    {"label": "r1", "value": "r1"},
    {"label": "r2", "value": "r2"},
    {"label": "t1", "value": "t1"},
    {"label": "t2", "value": "t2"},
    {"label": "rt", "value": "rt"},
]
page_1 = html.Div(style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-evenly', 'align-items': 'center'}, children=[
                    # First row
                    html.Div(style={'width': '100%', 'text-align': 'center'}, children=[html.H1(children=['Battery Modelling using the probabilistic programming language ',  html.Span('Turing.jl', style={'color': 'red', 'font-style': 'italic'}), ' and Cloud Analytics Platform ', html.Span('withdata', style={'color': 'red', 'font-style': 'italic'})], style={'vertical-align': 'top', 'color': 'purple'})]),

                    # second row: Left column
                    html.Div(style={'width': '35%'}, children=[html.Img(id="PK_model_diagram", src=battery_model_diagram_data, alt="my image", width="700", height="500", className="img_class", style={'text-align': 'right'})]),

                    # second row: middle column
                    html.Div(style={'width': '40%'}, children=[html.H3("Equivalent Circuit Model (ECM) [1,3]", style={'text-align': 'center', 'font-style': 'italic'}),
                                                               html.Img(id="battery_model_equation", src=battery_model_equation_data, alt="my image", width="750", height="520", className="img_class", style={'text-align': 'center'})]),
                    
                    # second row: right column
                    html.Div(style={'width': '20%', 'text-align': 'center'}, children=[
                        html.Label('rt', style={'text-align': 'right'}),
                        html.Div( dcc.Slider(id="rt",min=0,max=10,step=0.1,value=rt,marks={i: f"{i}" for i in range(0, 11, 2)})),
                        html.Label('r1', style={'text-align': 'right'}),
                        html.Div(dcc.Slider(id="r1",min=-5,max=5,step=0.1,value=r1,marks={i: f"{i}" for i in range(-5, 6, 2)})),
                        html.Label('t1', style={'text-align': 'right'}),
                        html.Div(dcc.Slider(id="t1",min=-5,max=5,step=0.1,value=t1,marks={i: f"{i}" for i in range(-5, 6, 2)})),
                        html.Label('r2', style={'text-align': 'right'}),
                        html.Div(dcc.Slider(id="r2",min=-5,max=5,step=0.1,value=r2,marks={i: f"{i}" for i in range(-5, 6, 2)})),
                        html.Label('t2', style={'text-align': 'right'}),
                        html.Div(dcc.Slider(id="t2",min=-5,max=5,step=0.1,value=t2,marks={i: f"{i}" for i in range(-5, 6, 2)})),
                        html.Label('noise level', style={'text-align': 'right'}),
                        html.Div(dcc.Slider(id="noise_std",min=0,max=1,step=0.01,value=noise_std,marks={i: f"{i}" for i in np.linspace(0, 1, num=5)}), style={'padding-bottom': '20px'}),
                    ]),
                    
                    
                    # third row: left column
                    html.Div(style={'width': '33%', 'text-align': 'center'}, children=[
                            html.Div([
                                html.H3("Real vs Imaginary Impedance", style={'text-align': 'center', 'font-style': 'italic'}),
                                dcc.Graph(id="real-im-plot"),
                            ], style={"width": "80%", "height": "300px", 'margin-bottom': '200px'}),
                        ]),
                    # third row: right column
                    html.Div(style={'width': '33%'}, children=[
                        html.Div([
                            html.H3("Real & Imaginary Impedance vs Log Frequency", style={'text-align': 'center', 'font-style': 'italic'}),
                            dcc.Graph(id="frequency-plot"),
                        ], style={"width": "80%", "height": "300px", 'margin-bottom': '200px'}),
                    ]),
                    # plot one param vs remaining life
                    html.Div(style={'width': '33%'}, children=[
                        html.H3("Lifespan vs parameter configurations", style={'text-align': 'center', 'font-style': 'italic'}),
                        html.Div(style={'width': '100%', 'display': 'flex'}, children=[
                        dcc.Dropdown(
                            id="dropdown",
                            options=dropdown_options,
                            value="r1",
                            style={"width": "50%"}
                        ),
                        html.Div(style={'width': '50%'}, children=[html.Div('Current remaining life:', style={"width": "70%"}), html.Div(id='life-box', children=5, style={'text-align': 'center', 'font-size': '21px', "width": "30%"})]),
                        ]),
                        html.Div([
                            dcc.Graph(id="life-plot", figure=px.line(x=r1_simulated_vec, y=r1_remaining_life_vec, labels={'x':'r1', 'y':'lifespan'}))
                        ], style={"width": "80%", "height": "300px", 'margin-bottom': '200px'}),            
                    ]),

                    html.Div(style={'height': '100px'}), # space div: doing nothing

                    ## the probabilistic model
                    html.Div(style={'width': '100%', 'text-align': 'center'}, children=[        
                        html.H2(['Probabilistic Modelling using ', html.I('Turing.jl', style={'color': 'red'})], style={'text-align': 'center'}),
                        html.Img(id="probabilistic_model", src=probabilistic_model_data, alt="my image", width="800", height="500", className="img_class"),
                        ]),

                    ## inference results: no callbacks
                    html.H2([html.I('withdata', style={'color': 'red'}), ' cloud-based data analytics platform: checkpoint-based continuous sampling'], style={'width': '100%','text-align': 'center'}),
                    # histograms
                    html.H3('From Cloud to Local: returning inference results', style={'width': '100%','text-align': 'center'}),
                    html.Div(style={'width': '100%', 'display': 'flex'}, children=[
                        html.Div(style={'width': '33.33%', 'text-align': 'right'}, children=[
                            dcc.Graph(id='hist-rt', figure=px.histogram(chain_dict['rt'], nbins=50, histnorm="probability density", marginal="box", title="rt Posterior Histogram"))
                        ]),
                        html.Div(style={'width': '33.33%', 'text-align': 'right'}, children=[
                            dcc.Graph(id='hist-r1', figure=px.histogram(chain_dict['r1'], nbins=50, histnorm="probability density", marginal="box", title="r1 Posterior Histogram"))
                        ]),
                        html.Div(style={'width': '33.33%', 'text-align': 'right'}, children=[
                            dcc.Graph(id='hist-t1', figure=px.histogram(chain_dict['t1'], nbins=50, histnorm="probability density", marginal="box", title="t1 Posterior Histogram"))
                        ]),
                    ]),
                    html.Div(style={'width': '100%', 'display': 'flex'}, children=[
                        html.Div(style={'width': '33.33%', 'text-align': 'right'}, children=[
                            dcc.Graph(id='hist-r2', figure=px.histogram(chain_dict['r2'], nbins=50, histnorm="probability density", marginal="box", title="r2 Posterior Histogram"))
                        ]),
                        html.Div(style={'width': '33.33%', 'text-align': 'right'}, children=[
                            dcc.Graph(id='hist-t2', figure=px.histogram(chain_dict['t2'], nbins=50, histnorm="probability density", marginal="box", title="t2 Posterior Histogram"))
                        ]),
                        html.Div(style={'width': '33.33%', 'text-align': 'right'}, children=[
                            dcc.Graph(id='hist-noise_std', figure=px.histogram(chain_dict['noise_std'], nbins=50, histnorm="probability density", marginal="box", title="noise std Posterior Histogram"))
                        ]),
                    ]),
                    # traces
                    html.Div(style={'width': '100%', 'display': 'flex'}, children=[
                        html.Div(style={'width': '33.33%', 'text-align': 'right'}, children=[
                            dcc.Graph(id='trace-rt', figure=px.line(pd.DataFrame(chain_dict['rt']), y=0, title="rt Trajectory"))
                        ]),
                        html.Div(style={'width': '33.33%', 'text-align': 'right'}, children=[
                            dcc.Graph(id='trace-r1', figure=px.line(pd.DataFrame(chain_dict['r1']), y=0, title="r1 Trajectory"))
                        ]),
                        html.Div(style={'width': '33.33%', 'text-align': 'right'}, children=[
                            dcc.Graph(id='trace-t1', figure=px.line(pd.DataFrame(chain_dict['t1']), y=0, title="t1 Trajectory"))
                        ]),
                    ]),
                    html.Div(style={'width': '100%', 'display': 'flex'}, children=[
                        html.Div(style={'width': '33.33%', 'text-align': 'right'}, children=[
                            dcc.Graph(id='trace-r2', figure=px.line(pd.DataFrame(chain_dict['r2']), y=0, title="r2 Trajectory"))
                        ]),
                        html.Div(style={'width': '33.33%', 'text-align': 'right'}, children=[
                            dcc.Graph(id='trace-t2', figure=px.line(pd.DataFrame(chain_dict['t2']), y=0, title="t2 Trajectory"))
                        ]),
                        html.Div(style={'width': '33.33%', 'text-align': 'right'}, children=[
                            dcc.Graph(id='trace-noise_std', figure=px.line(pd.DataFrame(chain_dict['noise_std']), y=0, title="noise std Trajectory"))
                        ]),
                    ]),                    
                ])

page_2 = html.Div(style={'display': 'flex', 'flex-wrap': 'wrap', 'width': '100%', 'justify-content': 'center'}, children=[        
                    html.Div(style={'width': '100%', 'text-align': 'center'}, children=[        
                    html.H1('Model Details', style={'vertical-align': 'top', 'color': 'purple'}),
                    html.Img(id="model-details", src=PK_model_details1_data, alt="my image", width="1200", height="600", className="img_class"),
                    ]),
                ])

page_3 = html.Div(style={'display': 'flex', 'flex-wrap': 'wrap', 'width': '100%', 'justify-content': 'center'}, children=[        
                    # Model details row 1    
                    html.Div(style={'width': '100%', 'text-align': 'center'}, children=[        
                    html.H1('Battery Management System [3]', style={'vertical-align': 'top', 'color': 'purple'}),
                    html.Img(id="BMS1", src=BMS_data1, alt="my image", width="1200", height="800", className="img_class"),
                    html.Img(id="BMS2", src=BMS_data2, alt="my image", width="1200", height="800", className="img_class"),
                    ]),
                ])

page_4 = html.Div(style={'display': 'flex', 'flex-wrap': 'wrap', 'width': '100%', 'justify-content': 'center'}, children=[        
                    html.Div(style={'width': '100%', 'text-align': 'center'}, children=[        
                    html.H1('Lifespan Estimation', style={'vertical-align': 'top', 'color': 'purple'}),
                    html.Img(id="references", src=battery_lifespan_data, alt="my image", width="1300", height="800", className="img_class"),
                    ]),
                ])

page_dropdown_options = [
    {'label': 'Page 1', 'value': 'page-1'},
    {'label': 'Page 2', 'value': 'page-2'},
    {'label': 'Page 3', 'value': 'page-3'},
    {'label': 'Page 4', 'value': 'page-4'}
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
    elif value == 'page-4':
        return page_4

# Define the callback for updating the plot in page_1 when the slider values change
@app.callback(
    [Output("real-im-plot", "figure"), Output("frequency-plot", "figure"), Output("life-plot", "figure"), Output("life-box", "children")],
    [Input("rt", "value"),
    Input("r1", "value"),
    Input("t1", "value"),
    Input("r2", "value"),
    Input("t2", "value"),
    Input("noise_std", "value"),
    Input("dropdown", "value")],
    )
def update_plot(rt, r1, t1, r2, t2, noise_std, dropdown):
    twoRCs_model_obj = towRCs_ECM(rt, r1, t1, r2, t2, omega=angular_f)
    noise = np.random.normal(0, noise_std, n)
    reals_simulated = twoRCs_model_obj.real_part() + noise
    ims_simulated = twoRCs_model_obj.imaginary_part() + noise
    
    remaining_life = round(calculate_remaining_life(rt, r1, t1, r2, t2), 2)

    fig1 = px.scatter(x=reals_simulated, y=ims_simulated) #title="Real vs Imaginary parts"
    fig1.update_xaxes(title_text="Re[Z]")
    fig1.update_yaxes(title_text="Im[Z]")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=np.log10(twoRCs_model_obj.omega / (2 * np.pi)),
                            y=ims_simulated, mode="markers", name="im"))
    fig2.add_trace(go.Scatter(x=np.log10(twoRCs_model_obj.omega / (2 * np.pi)),
                            y=reals_simulated, mode="markers", name="re"))
    fig2.update_xaxes(title_text="Log f")
    fig2.update_yaxes(title_text="Impedance")

    if dropdown == "r1":
        fig3 = px.line(x=r1_simulated_vec, y=r1_remaining_life_vec, labels={'x': 'r1', 'y': 'remaining life'})
    if dropdown == "r2":
        fig3 = px.line(x=r2_simulated_vec, y=r2_remaining_life_vec, labels={'x': 'r1', 'y': 'remaining life'})
    if dropdown == "t1":
        fig3 = px.line(x=t1_simulated_vec, y=t1_remaining_life_vec, labels={'x': 'r1', 'y': 'remaining life'})
    if dropdown == "t2":
        fig3 = px.line(x=t2_simulated_vec, y=t2_remaining_life_vec, labels={'x': 'r1', 'y': 'remaining life'})
    if dropdown == "rt":
        fig3 = px.line(x=rt_simulated_vec, y=rt_remaining_life_vec, labels={'x': 'r1', 'y': 'remaining life'})
    fig3.update_xaxes(title_text="parameter configuration")
    fig3.update_yaxes(title_text="lifespan")

    return fig1, fig2, fig3, remaining_life

if __name__ == '__main__':
    app.run_server('0.0.0.0', port=8051, debug=True)