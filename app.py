# -*- coding: utf-8 -*- 

# !pip install dash
# # !pip install dash==1.19.0
# !pip install jupyter_dash
# !pip install --upgrade plotly
# !pip install dash --upgrade

"""<!--  -->"""

# Import required libraries
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from jupyter_dash import JupyterDash
import plotly.graph_objects as go
import plotly.express as px
from dash import no_update
import dash_bootstrap_components as dbc


#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.linear_model import LinearRegression

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff

import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier

tcouleur = 'plotly_dark'
bcouleur = 'navy'
fcouleur = 'white'
fsize = 20


def plot_history_dash(df,feat): 
    fig_cm = px.histogram(data_frame= df,x=feat,opacity= 0.7)
     
    fig_cm.update_layout(
       barmode='overlay',
        # paper_bgcolor=bcouleur,  # Set the background color here
        font=dict(color=fcouleur,size=fsize),  # Set the font color to here 
        title_x=0.5,
        title_y=0.9,
        template=tcouleur
    )
    return fig_cm

def plot_history_all_dash(df ): 
    fig  = px.histogram(data_frame= df,opacity= .7)
    fig .update_layout(
       barmode='overlay',
        # paper_bgcolor=bcouleur,  # Set the background color here
        font=dict(color=fcouleur,size=fsize),  # Set the font color to here 
        title_x=0.5,
        title_y=0.9, 
        template=tcouleur
    )
    fig.update_xaxes( 
            title_font = {"size": 14},
            title_standoff = 25)
    return fig 

def plot_confusion_matrix_dash(y,y_predict,cmLabel):
    cm = confusion_matrix(y, y_predict)
    fig = ff.create_annotated_heatmap(cm, x=cmLabel, y=cmLabel, colorscale='Viridis')
    fig.update_layout(
        title='Confusion Matrix', 
        font=dict(color=fcouleur,size=fsize),  # Set the font color to here 
        template=tcouleur
    )
    fig.update_layout(
        # paper_bgcolor=bcouleur,  # Set the background color here
        font=dict(color=fcouleur,size=fsize),  # Set the font color to here 
        title_x=0.5,
        title_y=0.9, 
        template=tcouleur
    )
    fig.update_xaxes(
            side='bottom',
            title_text='Predicted labels',
            title_font = {"size": 18},
            title_standoff = 25)
    fig.update_yaxes(
            title_text = 'True labels',
            title_font = {"size": 18},
            title_standoff = 25)
    return fig

def plot_classification_report_dash(y, y_predict,cmLabel):

    report_str = classification_report(y, y_predict,  zero_division=0)
    report_lines = report_str.split('\n')

    # Remove empty lines
    report_lines = [line for line in report_lines if line.strip()]
    data = [line.split() for line in report_lines[1:]]
    colss = ['feature', 'precision',   'recall',  'f1-score',   'support', 'n1one']

    # Convert to a DataFrame
    report_df = pd.DataFrame(data, columns = colss )
    report_df = report_df[report_df.columns[:-1]]
    cm = report_df.iloc[:-3,1:].apply(pd.to_numeric).values

    colss1 = [  'precision',   'recall',  'f1-score',   'support']
    fig_cm = ff.create_annotated_heatmap(cm, x = colss1, y = cmLabel, colorscale='Viridis')
    fig_cm.update_layout(
          title='Classification Report',
          # paper_bgcolor=bcouleur,  # Set the background color here
          font=dict(color=fcouleur,size=fsize),  # Set the font color to here  
          title_x=0.5,
          title_y=0.9,
          template=tcouleur
      )
    return fig_cm

import pandas as pd
import zipfile
import requests
from io import BytesIO

# Define the URL of the ZIP archive
zip_url = 'http://archive.ics.uci.edu/static/public/53/iris.zip'
url =  'https://archive.ics.uci.edu/dataset/53/iris'
# Send a GET request to the URL to fetch the ZIP archive content
response = requests.get(zip_url)

# Check if the request was successful
if response.status_code == 200:
    # Read the content of the ZIP archive
    zip_data = BytesIO(response.content)

    # Create a ZipFile object from the fetched data
    with zipfile.ZipFile(zip_data, 'r') as zip_file:
        # List the files in the ZIP archive (to see the available files)
        file_list = zip_file.namelist()
        print("Files in ZIP archive:", file_list)

        # Assuming you want to read a specific CSV file from the ZIP archive
        csv_file_name = 'iris.data'

        # Check if the CSV file exists in the ZIP archive
        if csv_file_name in file_list:
            # Read the CSV file into a DataFrame
            with zip_file.open(csv_file_name) as csv_file:
                 df = pd.read_csv(csv_file, encoding="ISO-8859-1")

        else:
            print(f"CSV file '{csv_file_name}' not found in the ZIP archive.")
else:
    print("Failed to fetch the ZIP archive.")

cols = ["slength", "swidth", "plength","pwidth","class"]
df.columns =  cols
df.head()

unique_values = df['class'].unique()
print(unique_values)

mapping = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2 }
df["class"] = df["class"].map(mapping)
cmLabel = list(mapping.keys())# ['setosa', 'versicolor','virginica']
df.head()

from requests.api import head
filtered_df = df[df.columns[0:-1]]
filtered_df.head( )

X = df[df.columns[:-1]].values
Y = df[df.columns[-1]].values

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.3, random_state=4)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_DT = clf.predict(X_test)

from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression(solver='lbfgs', max_iter=1000)
lg_model = lg_model.fit(X_train, y_train)
y_pred_LG = lg_model.predict(X_test)

from sklearn.svm import SVC
svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)
y_pred_SVC = svm_model.predict(X_test)
# plot_confusion_matrix(y_test,y_pred_SVC)

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_KNN = knn_model.predict(X_test)

from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)
y_pred_NB = nb_model.predict(X_test)

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=2500, tol=1e-3, penalty = 'elasticnet'))
clf.fit(X_train, y_train)
y_pred_SGD = clf.predict(X_test)



dropdown_options_style = {'color': 'white', 'background-color' : 'gray'}
dropdown_options = [
    {'label': 'All Features', 'value': 'ALL', 'style': dropdown_options_style}
]

for col in cols[:-1]:
    dropdown_options.append({'label': col, 'value': col, 'style':  dropdown_options_style})


# Create a dash application Cyborg

app =  dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
JupyterDash.infer_jupyter_proxy_config()

server = app.server
app.config.suppress_callback_exceptions = True
 
app.layout = html.Div(
    style={
        'color' : 'black',
        'backgroundColor': 'black',  # Set the background color of the app here
        'height': '100vh'  # Set the height of the app to fill the viewport
    },  
     
   children=[
    html.H1('Iris Dataset Analysis', style={'textAlign': 'center', 'color': 'white', 'background-color' : 'black',   'font-size': 40}),
    html.Div([
    html.Br(),
    html.Div(['Analysis of the Iris dataset and evaluation of various classification machine',
              html.Br(),
            'learning algorithms used on data taken from ', 
            html.Br(),
            dcc.Link(url, href=url, target="_blank")],
            style={'textAlign': 'center', 'color': 'white', 'background-color' : 'black',   'font-size': 30}), 
            html.Br(),
            ]), 

          # Create an outer division
            html.Div([
                html.Div([
                  dcc.Dropdown(
                      id='site-dropdown1',
                      options=dropdown_options,#[
                      #         {'label': 'All Features', 'value': 'ALL',style{'color':'white'}},
                      #         {'label': cols[0], 'value': cols[0]},
                      #         {'label': cols[1], 'value': cols[1]},
                      #         {'label': cols[2], 'value':  cols[2]}
                      #         ],
                      value='ALL',
                      placeholder='Select a feature',
                      style={
                            'width':'80%',
                            'padding':'3px',
                            'font-size': '20px',   
                            'text-align-last' : 'center' ,
                            'margin': 'auto' , # Center-align the dropdown horizontally
                            'background' : 'black', 
                            'color': 'black', 
                            },
                      searchable=True
                  ) ,
                  html.Div(id='output-graph1') ,
                ]),
                html.Div([
                  dcc.Dropdown(
                      id='site-dropdown2',
                      options=[
                              {'label':  'Logistic Regression',          'value': 'LG',   'style':  dropdown_options_style},
                              {'label': 'Decision Tree Classifier',      'value': 'DT',   'style':  dropdown_options_style},
                              {'label': 'K-Nearest Neighbors',           'value': 'KNN',  'style':  dropdown_options_style },
                              {'label': 'Support Vector Classification', 'value': 'SVC',  'style':  dropdown_options_style},
                              {'label': 'Gaussian Naive Bayes',          'value': 'NB',   'style':  dropdown_options_style},
                              {'label': 'Stochastic Gradient Descent',   'value': 'SGD' , 'style':  dropdown_options_style}
                              ],
                      value='LG',
                      placeholder='Select a Machine Learning Classifier',                      
                      style={
                            'width':'80%',
                            'padding':'3px',
                            'font-size': '20px', 
                            'text-align-last' : 'center' ,
                            'margin': 'auto',  # Center-align the dropdown horizontally
                            'background-color' : 'black', 
                            'color': 'black'
                            },
                      searchable=True, 
                  ) ,
        html.Div([
            html.Div(id='output-graph2', style={'width': '50%', 'display': 'inline-block'}),
            html.Div(id='output-graph3', style={'width': '50%', 'display': 'inline-block'}),
        ]),
        ]),
        ]), 
    ]
    )
# Function decorator to specify function input and output
@app.callback(
    [
     Output('output-graph1', 'children'),
     Output('output-graph2', 'children'),
     Output('output-graph3', 'children')],
      [Input('site-dropdown1', 'value'),
       Input('site-dropdown2', 'value')]
)

def update_graph(feature,ml):
  filtered_df = df[df.columns[0:-1]]

  if feature == 'ALL':
    figure1 =  dcc.Graph( figure = plot_history_all_dash(filtered_df ) )
  else:
    figure1 =  dcc.Graph( figure = plot_history_dash(filtered_df,feature) )

  if ml == 'LG' :
    fig2 =   dcc.Graph( figure = plot_confusion_matrix_dash(y_test,y_pred_LG,cmLabel))
    fig3 =   dcc.Graph( figure = plot_classification_report_dash(y_test,y_pred_LG,cmLabel))
  elif ml == 'DT':
    fig2 =   dcc.Graph( figure = plot_confusion_matrix_dash(y_test,y_pred_DT,cmLabel))
    fig3 =   dcc.Graph( figure = plot_classification_report_dash(y_test,y_pred_DT,cmLabel))
  elif ml == 'KNN':
    fig2 =   dcc.Graph( figure = plot_confusion_matrix_dash(y_test,y_pred_KNN,cmLabel))
    fig3 =   dcc.Graph( figure = plot_classification_report_dash(y_test,y_pred_KNN,cmLabel))
  elif ml == 'SVC':
    fig2 =   dcc.Graph( figure = plot_confusion_matrix_dash(y_test,y_pred_SVC,cmLabel))
    fig3 =   dcc.Graph( figure = plot_classification_report_dash(y_test,y_pred_SVC,cmLabel))
  elif ml == 'NB':
    fig2 =   dcc.Graph( figure = plot_confusion_matrix_dash(y_test,y_pred_NB,cmLabel))
    fig3 =   dcc.Graph( figure = plot_classification_report_dash(y_test,y_pred_NB,cmLabel))
  elif ml == 'SGD':
    fig2 =   dcc.Graph( figure = plot_confusion_matrix_dash(y_test,y_pred_SGD,cmLabel))
    fig3 =   dcc.Graph( figure = plot_classification_report_dash(y_test,y_pred_SGD,cmLabel))

  return  [figure1,fig2,fig3]

# Run the app
if __name__ == '__main__':
    # REVIEW8: Adding dev_tools_ui=False, dev_tools_props_check=False can prevent error appearing before calling callback function
    app.run_server(  host="localhost" , debug=False)#, dev_tools_ui=False, dev_tools_props_check=False)
