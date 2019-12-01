import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import dash_table
from dash.dependencies import Input, Output, State
import pickle
from src.TSP import TSP
from imblearn.over_sampling import SMOTE

l_l =pd.read_csv('lat_lng.csv')

df=pd.DataFrame({'customer_lat':['Please' for i in range(1000)],
    'customer_lng':['Generate' for i in range(1000)],
    'seller_lat':['Data' for i in range(1000)],
    'seller_lng':['First' for i in range(1000)],
    'cluster':['!' for i in range(1000)]
}).reset_index()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children = [
    html.H1('Warehouse Placement Simulator'),
    html.P('Created by: Gusti Adli Anshari'), 
    html.Div(className='row homemade',children=[
        html.Div(className='col-1'),
        html.Div(className='col-10',children=[
            html.Div(className='row homemade',children=[
                html.Strong(className='col-4',children='Select Warehouse Quantity')
            ]),
            html.Center(dcc.Slider(
                id='n-slider',
                min=1,
                max=8,
                step=1,
                marks={i:str(i) for i in range(1,9,1)},
                value=4,
            ))
        ]),
        html.Div(className='col-1')
    ]),
    html.Div(className='row center',children=[
        html.Div(children=[
            html.Img(
                className='figure',
                id='cluster-map',
            )
        ])
    ]),
    html.Div(className='row',children=[
        html.Div(className='col-1'),
        html.Div(className='col-3',children=[
            html.Div(className='column',children=[
                html.P(
                    className='row',
                    children='Number of Warehouses'
                ),
                dcc.Input(
                    className='row',
                    id='n-cluster',
                    type='number',
                    min=1,max=8,step=1
                )
            ]),
        ]),
        html.Div(className='col-3',children=[
            html.Div(className='column',children=[
                html.P(
                    className='row',
                    children='Number of Orders'
                ),
                dcc.Input(
                    className='row',
                    id='n-order',
                    type='number',
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
        html.Div(className='col-3'),
        html.Div(className='col-6 center',children=[dash_table.DataTable(
            id='cluster-info',
            data=pd.DataFrame({'Cluster':['No'],'Quantity':['Information'],'Average_Distance':['Available']}).to_dict('records'),
            columns=[{'id':i,'name':i} for i in ['Cluster','Quantity','Average_Distance']],
        )]),
        html.Div(className='col-3')
    ]),
    html.Div(className='row',children=[
        html.Div(className = 'col-12',children = dcc.Graph(
            id = 'map2',
            config=dict(mapboxAccessToken='pk.eyJ1IjoiZ3N0ZGwiLCJhIjoiY2szOHpvcGM4MGJ3MDNibDMwNWVnam81ZSJ9.UDewXUFso2Tb9S3OlWfsmg'),
            figure = {
                'data': [go.Scattermapbox(dict(
                ))
                ],
                'layout':dict(
                    hovermode='closest',
                    mapbox=go.layout.Mapbox(
                        bearing=0,
                        center=go.layout.mapbox.Center(
                            lat=-15,
                            lon=-56
                        ),
                        pitch=0,
                        zoom=1.5
                    )
                )  
            },
        ))
    ]),
    html.Div(children=[
        html.Center(children=[
            html.H3('Max Delivering Quantity'),
            dcc.Input(
                id='simulate-iter',
                type='number',
                value=2,
                min=2,max=1000000000,step=1
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
    return app.get_asset_url(f'km3_iter{n-1}.jpg')

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
            labels=pickle.load(open(f'model_data_dump/km3_iter{cluster}.label','rb'))
            X,y=SMOTE().fit_resample(l_l,labels)
            lat_lng=pd.DataFrame(X,columns=l_l.columns)
            c_lat,c_lng,s_lat,s_lng,clust=[],[],[],[],[]
            while len(c_lat)<order:
                ac_clus,as_clus=1,2
                while ac_clus!=as_clus:
                    idx_ac=np.random.randint(0,len(lat_lng)-1)
                    idx_as=np.random.randint(0,len(lat_lng)-1)
                    ac_lat,ac_lng=lat_lng.loc[idx_ac,'lat'],lat_lng.loc[idx_ac,'lng']
                    as_lat,as_lng=lat_lng.loc[idx_as,'lat'],lat_lng.loc[idx_as,'lng']
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
def get_cluster_map(n_clicks,df,cluster):
    if n_clicks==0:
        raise dash.exceptions.PreventUpdate
    else:
        df=pd.DataFrame(df)
        labels=df['cluster'].unique()
        centroids=pickle.load(open(f'model_data_dump/km3_iter8.ctr','rb'))
        scatter=[]
        for label in labels:
            scatter.append(go.Scattermapbox(dict(
                lat=[centroids[label,1]],
                lon=[centroids[label,0]],
                text=f'Cluster Center#{label}',
                name=f'Cluster Center#{label}',
                marker={'color':'blue','symbol':'star'}
            )))
            for user,color in zip(['seller','customer'],['green','red']):
                scatter.append(go.Scattermapbox(dict(
                    lat=[i for i in df[df['cluster']==label][user+'_lat']],
                    lon=[i for i in df[df['cluster']==label][user+'_lng']],
                    text=user+'(cluster:'+str(label)+')',
                    name=user+'(cluster:'+str(label)+')',
                    marker={'color':color}
                )))

        figure = {
                'data':scatter,
                'layout':dict(
                    hovermode='closest',
                    mapbox=go.layout.Mapbox(
                        bearing=0,
                        center=go.layout.mapbox.Center(
                            lat=-15,
                            lon=-56
                        ),
                        pitch=0,
                        zoom=1.5
                    )
                )  
            }
        return figure

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
            s=[
                *s,
                html.H1(html.Strong(f'Details for warehouse {cluster}:')),
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
        s=[
            *s,
            html.H2(f'TOTAL DISTANCE TRAVELED PICKING UP ITEMS = {sum(list(sel_dist.values()))}'),html.P(),html.P(),
            html.H2(f'TOTAL DISTANCE TRAVELED DELIVERING ITEMS = {sum(list(cus_dist.values()))}'),html.P(),html.P()
        ]
        return s

if __name__ == '__main__':
    app.run_server(debug=True)