# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Nemanja Hranisavljevic
# Contact: nemanja@ai4cps.com


import base64
from selfx.dash.routing_utils import construct_id, construct_url, ROUTE_PREFIX
from dash import dcc, html, State, Input, Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import mlflow


def get_sidebar(app, system, user, feature, start, end):
    features = app.features[system].get(user, dict())
    nav_children = []
    for feature_name, feature_class in features.items():
        feature_obj = feature_class()
        nav_link_children = [html.I(feature_obj.icon(), className="material-icons"),
                             feature_name]
        new_link = dbc.NavLink(children=nav_link_children,
                               href=construct_url(system, user, feature_name, start, end),
                               id=construct_id(system, user, feature_name), active=feature_name == feature,
                               external_link=True, className="sidebar_buttons_style")
        nav_children.append(new_link)

    content = [dbc.Nav(children=nav_children, vertical=True, pills=True, id=construct_id('nav_tools'))]

    sidebar = html.Div(content, className="sidebar_style")
    return sidebar


def get_topbar(selfx, systems, roles, logo, date_picker, system, role, feature, start, end):
    topbar_elements_left = []
    topbar_elements_right = []
    topbar_elements = []

    style = {} if date_picker else {'display': 'none'}
    topbar_elements_right.append(dcc.DatePickerRange(
        id='date-picker', persistence=False, start_date=start.replace('_', '.'), style=style,
        end_date=end.replace('_', '.'), display_format='DD.MM.YYYY', minimum_nights=0, updatemode='bothdates'))
    topbar_elements_right.append(html.Button(className="reevaluate_button",
                                                         id=construct_id("reevaluate"),
                                                         children=[html.I('replay', className="material-icons"),
                                                                   'Reevaluate']))

    for_plant_dropdown = [{"label": sn, "value": sn} for sn in systems]

    selfxlogo = selfx.app.get_asset_url('Logo_SelfX.svg')

    if not logo:
        topbar_elements.append(html.Div(html.A(html.Img(className='logo', src=selfxlogo), href=ROUTE_PREFIX[:-1]),
                                        className="logo_div"))
    else:
        topbar_elements = []

        for l in logo:
            # Read and encode the image
            with open(l, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('ascii')

            if l.endswith('svg'):
                src = 'data:image/svg+xml;base64,{}'.format(encoded_image)
            else:
                src = 'data:image/png;base64,{}'.format(encoded_image)
            topbar_elements.append(html.Div(html.A(html.Img(className='logo', src=src), href=ROUTE_PREFIX[:-1]), className="logo_div"))
        # topbar_elements_left.append(html.Label("Plant:", className="dropdownLabel"))

    dd_elements = []
    for el in for_plant_dropdown:
        dd_elements.append(dbc.DropdownMenuItem(el['label'],
                                                href=construct_url(el['label'], role, feature, start, end)))
    dropdown = dbc.DropdownMenu(children=dd_elements, label=system, id=construct_id('plant_dropdown'), className='plant_dropdown')
    # dcc.Dropdown(id=construct_id("plant_dropdown"),
    #              className='plant_dropdown',
    #              clearable=False,
    #              options=for_plant_dropdown,
    #              value=val,
    #              optionHeight=60, persistence=True)
    topbar_elements_left.append(dropdown)
    # topbar_elements_left.append(html.Label("User role:", className="dropdownLabel"))

    dd_elements = []
    for el in roles:
        dd_elements.append(dbc.DropdownMenuItem(el, href=construct_url(system, el, feature, start, end)))
    dropdown = dbc.DropdownMenu(children=dd_elements, label=role, id=construct_id('role_dropdown'), className='role_dropdown')
    topbar_elements_left.append(dropdown)

    # topbar_elements_left.append(dcc.Dropdown(options=[],
    #                                          clearable=False, value=None, searchable=False,
    #                                          id=construct_id('user_dropdown'), className='user_dropdown',
    #                                          optionHeight=60, persistence=True)),
    topbar_left = html.Div(children=topbar_elements_left, className="topbar_style_left")
    topbar_right = html.Div(children=topbar_elements_right, className="topbar_style_right")
    topbar_elements.append(html.Div([topbar_left, topbar_right], className="topbar_style_right_of_logo"))
    return html.Div(children=topbar_elements, className="topbar_style")


def machine_learning(app=None, method="", **kwargs):
    content = []
    for k, choice in kwargs.items():
        content.append(html.Div([
        html.Label(f'{k}:'),
        dcc.Dropdown(id=k+method,
                     options=[{'label': n, 'value': n} for n in choice], persistence=True,
                     value=choice[0], clearable=False, multi=False)], className="content_control_wide"))
        content.append(html.Br())

    content.append(html.Div([html.Label(''),
                  html.Button('Train', id='train_btn'+method, style={'width':100, 'text-align': 'center'})],
                 className="content_control_wide"))

    runs = mlflow.search_runs()
    exists_model = not runs.empty and 'tags.mlflow.log-model.history' in runs.columns and \
                   'params.algorithm' in runs.columns
    if exists_model:
        runs = runs[~runs['tags.mlflow.log-model.history'].isnull()]

    options = [dict(label=f"{r['params.algorithm']} | {r['run_id']}", value=r['run_id']) for r in
               runs.to_dict('records') if (not method or r["params.algorithm"] == method)]

    # val = options[0]['value'] if len(options) > 0 else None
    val = options[0]['value'] if len(options) > 0 else None

    content += [html.H2('Evaluation'),
                html.Div([html.Label('Model:'),
                          dcc.Dropdown(options=options,
                                       value=val,
                                       clearable=False,
                                       searchable=True,
                                       id='model_search'+method)], className='content_control_wide'),
                html.Div(children=[], id="model_results"+method)]

    # if app is not None:
    #     @app.callback(Output('model_results', 'children'), Input('model_search', 'value'))
    #     def fun(model):
    #         try:
    #             run = mlflow.get_run(model)
    #         except:
    #             return 'Model could not be found.'
    #         model_history = run.data.tags['mlflow.log-model.history']
    #         model_history = json.loads(model_history)
    #         model_path = model_history[0]['artifact_path']
    #         model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/{model_path}")
    #         model = model.unwrap_python_model()
    #         return

    return content


def register_modal_state_transition_callbacks(dash_app, graph_id=None):
    callback_app = dash_app.app

    @callback_app.callback(Output(f'{graph_id}-modal-state-data', 'is_open'),
                           Output(f'{graph_id}-modal-state-data', 'children'),
                           Input(graph_id, 'tapNodeData'), prevent_initial_call=True)
    def render_content(data_node):
        text = data_node['id']
        return True, [dbc.ModalHeader("State"), dbc.ModalBody(html.Div(children=[text]))]

    @callback_app.callback(Output(f'{graph_id}-modal-transition-data', 'is_open'),
                           Output(f'{graph_id}-modal-transition-data', 'children'),
                           Input(graph_id, 'tapEdgeData'),
                           prevent_initial_call=True)
    def render_content(data_edge):
        if data_edge is not None:
            fig = None
            # fig = self.plot_transition(data_edge['source'], data_edge['target'])
            content = [data_edge['source'], html.Br(), data_edge['target'],
                       dcc.Graph(figure=fig)]
            print('return something')
            return True, [dbc.ModalHeader(f"Event: {data_edge['label']}"),
                          dbc.ModalBody(html.Div(children=content))]
        raise PreventUpdate