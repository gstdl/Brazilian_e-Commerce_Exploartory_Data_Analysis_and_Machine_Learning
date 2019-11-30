import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import dash_table
from dash.dependencies import Input, Output, State
import pickle
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib.pyplot import legend,gcf,subplots
from plotly.tools import mpl_to_plotly
from src.TSP import TSP
from imblearn.over_sampling import SMOTE

import dash_bootstrap_components as dbc
import dash_html_components as html

import geopandas as gpd
import mlrose

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
brazil = world[world['name']=='Brazil']
l_l =pd.read_csv('../lat_lng.csv')

df=pd.DataFrame({'customer_lat':['Please' for i in range(1000)],
    'customer_lng':['Generate' for i in range(1000)],
    'seller_lat':['Data' for i in range(1000)],
    'seller_lng':['First' for i in range(1000)],
    'cluster':['!' for i in range(1000)]
}).reset_index()

def brazilplot():
    brazil.plot()
    fig=gcf()
    return mpl_to_plotly(fig)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[*external_stylesheets,dbc.themes.BOOTSTRAP])

app.layout = html.Div(children = [
    html.H1('Warehouse Placement Simulator'),
    html.P('Created by: Gusti Adli Anshari'), 
    # content_style = {
    #     'fontFamily': 'Arial',
    #     'borderBottom': '1px solid #d6d6d6',
    #     'borderRight': '1px solid #d6d6d6',
    #     'borderLeft': '1px solid #d6d6d6',
    #     'padding': '44px'
    # },
    html.Div(className='row homemade',children=[
        html.Div(className='col-1'),
        html.Div(className='col-10',children=[
            html.Div(className='row homemade',children=[
                html.Strong(className='col-4',children='Select Warehouse Quantity'),
                # dcc.Input(
                #     className='col-3',
                #     id='n-input',
                #     type='number',
                #     value=6,
                #     min=1,max=100,step=1
                # )
            ]),
            html.Center(dcc.Slider(
                id='n-slider',
                min=1,
                max=100,
                step=1,
                marks={i:str(i) for i in range(1,101,10)},
                value=6,
            ))
        ]),
        html.Div(className='col-1')
    ]),
    html.Div(className='row center',children=[
        html.Div(children=[
            html.Img(
                className='figure',
                id='cluster-map',
                src=app.get_asset_url('KMeans6.jpg')
            )
        ])
    ]),
    html.Div(className='row',children=[
        html.Div(className='col-1'),
        html.Div(className='col-3',children=[
            html.Div(className='column',children=[
                # html.P(
                #     className='row',
                #     children='Number of Customers'
                # ),
                # dcc.Input(
                #     className='row margtop',
                #     id='n-customer',
                #     type='number',
                #     # value=6,
                #     min=1,max=100,step=1
                # ),
                html.P(
                    className='row',
                    children='Number of Warehouses'
                ),
                dcc.Input(
                    className='row',
                    id='n-cluster',
                    type='number',
                    # value=6,
                    min=1,max=100,step=1
                )
            ]),
        ]),
        html.Div(className='col-3',children=[
            html.Div(className='column',children=[
                # html.P(
                #     className='row',
                #     children='Number of Sellers'
                # ),
                # dcc.Input(
                #     className='row',
                #     id='n-seller',
                #     type='number',
                #     # value=6,
                #     min=1,max=100,step=1
                # ),
                html.P(
                    className='row',
                    children='Number of Orders'
                ),
                dcc.Input(
                    className='row',
                    id='n-order',
                    type='number',
                    # value=6,
                    min=1,max=1000000000,step=1
                )
            ])
        ]),
        html.Div(className='col-5',children=[
            html.Button(
                className='btn btn-primary margtop',
                id='data-generate', 
                n_clicks=0, 
                children='Generate Data',
            )
        ])
    ]),
    html.Div(className='row homemade',children=[
        html.Div(className='col-1'),
        html.Div(className='col-10',children=[dash_table.DataTable(
            id='data-generated',
            data=df.to_dict('records'),
            columns=[{'id':i,'name':i} for i in ['index','customer_lat','customer_lng','seller_lat','seller_lng','cluster']],
            page_size=10,
        )]),
        html.Div(className='col-1')
    ]),
    html.Div(className='column',children=[
        html.Center(html.Button(
            className='btn btn-primary homemade center',
            id='cluster-button',
            n_clicks=0,
            children='Get Generated Data Cluster Information'
        )),
        # html.Div(className='row'),
        html.Div(className='col-3'),
        html.Div(className='col-6 center',children=[dash_table.DataTable(
            id='cluster-info',
            data=pd.DataFrame({'Cluster':['No'],'Quantity':['Information'],'Average_Distance':['Available']}).to_dict('records'),
            columns=[{'id':i,'name':i} for i in ['Cluster','Quantity','Average_Distance']],
        )]),
        html.Div(className='col-3')
    ]),
    html.Div(className='row',children=[
        html.Div(children=[
            dcc.Graph(
                className='row center',
                id='map2',
                figure=brazilplot()
            )
        ])
    ]),
    html.Div(children=[
        html.Center(children=[
            html.H3('Max Delivering Quantity'),
            dcc.Input(
                id='simulate-iter',
                type='number',
                value=10,
                min=1,max=1000000000,step=1
            ),
            html.Div(className='col-5'),
            html.Button(
                className='btn btn-primary margtop',
                id='simulate-button', 
                n_clicks=0, 
                children='Simulate Data',
            ) 
        ]),
    ]),
    html.Div(className='col-12'),
    html.Div(style={
        'margin-top':'14px',
        'margin-bottom':'14px'
    },className='row',children=[
        html.Div(className='col-1'),
        html.Div(className='col-10',id='simulation-output',children=html.Center('Click The Button Above To Simulate Model')),
        html.Div(className='col-1')
    ])
    
],
)

@app.callback(
    Output(component_id='cluster-map',component_property='src'),
    [Input(component_id='n-slider',component_property='value')]
)
def n_slider1(n):
    return app.get_asset_url(f'KMeans{n}.jpg')

@app.callback(
    Output(component_id='data-generated',component_property='data'),
    [Input(component_id='data-generate',component_property='n_clicks')],
    [State(component_id=i,component_property='value') for i in ['n-cluster','n-order']]
)
def get_data(n_clicks,cluster,order):
    if n_clicks==0:
        raise dash.exceptions.PreventUpdate
    else:
        if order<cluster:
            df=pd.DataFrame({'customer_lat':['Other' for i in range(1000)],
                'customer_lng':['Values' for i in range(1000)],
                'seller_lat':['Must Be' for i in range(1000)],
                'seller_lng':['Larger Than' for i in range(1000)],
                'cluster':['Warehouse Quantity' for i in range(1000)]
            }).reset_index()
        else:
            # from random import randint
            km=pickle.load(open(f'../KMeans_{cluster}.model','rb'))
            X,y=SMOTE().fit_resample(l_l,km.labels_)
            lat_lng=pd.DataFrame(X,columns=l_l.columns)
            c_lat,c_lng,s_lat,s_lng,clust=[],[],[],[],[]
            while len(c_lat)<order:
                ac_clus,as_clus=1,2
                while ac_clus!=as_clus:
                    # # lat=-33.75116944<=cols['geolocation_lat']<=5.27438888
                    # # lng=-73.98283055<=cols['geolocation_lng']<=-34.79314722
                    # ac_lat=np.random.randint(-3375,527)/100
                    # ac_lng=np.random.randint(-7398,-3479)/100
                    # as_lat=np.random.randint(-3375,527)/100
                    # as_lng=np.random.randint(-7398,-3479)/100
                    idx_ac=np.random.randint(0,len(lat_lng)-1)
                    idx_as=np.random.randint(0,len(lat_lng)-1)
                    ac_lat,ac_lng=lat_lng.loc[idx_ac,'lat'],lat_lng.loc[idx_ac,'lng']
                    as_lat,as_lng=lat_lng.loc[idx_as,'lat'],lat_lng.loc[idx_as,'lng']

                    # clus=km.predict([[ac_lat,ac_lng],[as_lat,as_lng]])
                    # ac_clus=clus[0]
                    # as_clus=clus[1]
                    as_clus=y[idx_as]
                    ac_clus=y[idx_ac]
                    
                c_lat.append(ac_lat)
                c_lng.append(ac_lng)
                s_lat.append(as_lat)
                s_lng.append(as_lng)
                clust.append(as_clus)
            df=pd.DataFrame({'customer_lat':c_lat,
                'customer_lng':c_lng,
                'seller_lat':s_lat,
                'seller_lng':s_lng,
                'cluster':clust
            }).reset_index()
            df=pd.DataFrame(df)
        return df.to_dict('records')

@app.callback(
    Output(component_id='cluster-info',component_property='data'),
    [Input(component_id='cluster-button',component_property='n_clicks')],
    [State(component_id='data-generated',component_property='data')]
)
def get_cluster_info(n_clicks,df):
    if n_clicks==0:
        raise dash.exceptions.PreventUpdate
    else:
        df=pd.DataFrame(df)
        df['distance']=df[['seller_lat','seller_lng','customer_lat','customer_lng']].dropna().apply(lambda x:np.sqrt((x[0]-x[2])**2+(x[1]-x[3])**2),axis=1)
        d=df['cluster'].value_counts().to_frame().reset_index()
        d.rename(columns={i:j for i, j in zip(d.columns,['Cluster','Quantity','Average_Distance'])},inplace=True)
        d['Average_Distance']=d['Cluster'].apply(lambda x:df[df['cluster']==x]['distance'].mean())
        return d.to_dict('records')

@app.callback(
    Output(component_id='map2',component_property='figure'),
    [Input(component_id='cluster-button',component_property='n_clicks')],
    [State(component_id='data-generated',component_property='data'),
    State(component_id='n-cluster',component_property='value')]
)
def get_cluster_map(n_clicks,df,n):
    if n_clicks==0:
        raise dash.exceptions.PreventUpdate
    else:
        km=pickle.load(open(f'../KMeans_{n}.model','rb'))
        df=pd.DataFrame(df)
        fig,ax=subplots(1,1)
        brazil.plot(color='whitesmoke',edgecolor='black',figsize=(16,8),ax=ax)
        sns.scatterplot(df['seller_lng'],df['seller_lat'],ax=ax,color='green')
        sns.scatterplot(df['customer_lng'],df['customer_lat'],ax=ax,color='red')
        sns.scatterplot([i[0] for i in km.cluster_centers_],[i[1] for i in km.cluster_centers_],marker='*',s=500,ax=ax,color='b')
        legend(['Sellers','Customer','Warehouse'])
        return mpl_to_plotly(fig)

@app.callback(
    Output(component_id='simulation-output',component_property='children'),
    [Input(component_id='simulate-button',component_property='n_clicks')],
    [State(component_id='data-generated',component_property='data'),
    State(component_id='simulate-iter',component_property='value'),
    State(component_id='n-cluster',component_property='value')]
)
def ysp(n_clicks,df,stops,cluster):
    if n_clicks==0:
        raise dash.exceptions.PreventUpdate
    else:
        df=pd.DataFrame(df)
        sel_iter={}
        cus_iter={}
        cus_dist={}
        sel_dist={}
        cus_dic={}
        sel_dic={}
        cus_ord={}
        sel_ord={}
        s=[]
        for cluster in df['cluster'].unique():
            d=df[df['cluster']==cluster]
            sel_dic[cluster],sel_iter[cluster],sel_dist[cluster],sel_ord[cluster]=TSP(stops,[[d.loc[i,'seller_lat'],d.loc[i,'seller_lng']] for i in d.index])
            cus_dic[cluster],cus_iter[cluster],cus_dist[cluster],cus_ord[cluster]=TSP(stops,[[d.loc[i,'customer_lat'],d.loc[i,'customer_lng']] for i in d.index])
            s=[*s,html.H1(html.Strong(f'Details for warehouse {cluster}:')),
            html.H4(f'Distance Traveled Picking up Items = {sel_dist[cluster]}'),
            html.H4(f'Stops During Item Pick Up = {sel_iter[cluster]}'),
            html.P(),
            html.P('Details of distance traveled before each stop is shown below.'),
            html.P(f'Sequence= {sel_ord[cluster]}'),
            html.P(f' Distances= {sel_dic[cluster]}'),
            html.P(),html.P(),
            html.H4(f'Distance Traveled Delivering Items = {cus_dist[cluster]}'),
            html.H4(f'Stops During Item Delivery = {cus_iter[cluster]}'),
            html.P(f'Details of distance traveled before each stop is shown below.'),
            html.P(),
            html.P(f'Sequence= {cus_ord[cluster]}'),
            html.P(f' Distances= {cus_dic[cluster]}'),html.P(),html.P(),
            html.P()
            ]
        s=[*s,
        html.H2(f'TOTAL DISTANCE TRAVELED PICKING UP ITEMS = {sum(list(sel_dist.values()))}'),html.P(),html.P(),
        html.H2(f'TOTAL DISTANCE TRAVELED DELIVERING ITEMS = {sum(list(cus_dist.values()))}'),html.P(),html.P()
        ]
        return s

if __name__ == '__main__':
    app.run_server(debug=True)