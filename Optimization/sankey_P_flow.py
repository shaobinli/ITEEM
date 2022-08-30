

# imports
import pandas as pd
import numpy as np
import seaborn as sns
from ITEEM import ITEEM

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
from plotly.graph_objs import *
import plotly.io as pio
import plotly
# pio.orca.config.executable = r'C:\Users\Shaobin\anaconda3\pkgs\plotly-orca-1.3.1-1\orca_app\orca.exe'
pio.orca.config.executable = r'C:\Users\X1\anaconda3\pkgs\plotly-orca-1.3.1-1\orca_app\orca.exe'
init_notebook_mode()

'''color start'''
import plotly.graph_objects as go
import urllib, json

url = 'https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json'
response = urllib.request.urlopen(url)
data = json.loads(response.read())
# override gray link colors with 'source' colors
opacity = 0.4
# change 'magenta' to its 'rgba' value to add opacity
data['data'][0]['node']['color'] = ['rgba(255,0,255,0.8)' if color == "magenta" else color for color in data['data'][0]['node']['color']]
data['data'][0]['link']['color'] = [data['data'][0]['node']['color'][src].replace("0.8", str(opacity))
                                    for src in data['data'][0]['link']['source']]

color_node = data['data'][0]['node']['color']
color_node[1] = 'rgba(95,190,68,0.8)'; color_node[2] = 'rgba(101,105,111,0.8)';
color_link = data['data'][0]['link']['color']
# color_node2 = list(set(color_node))
# color_node3 = []
# source = [0, 1, 1, 1, 1, 1, 2, 2, 2,  3, 3, 3, 3,  4, 4, 5,  5,  5,  6]

# for i in source:
#     color_node3.append(color_node[i])
# color_link3 = [i.replace('0.8', '0.4') for i in color_node3]

b = 'rgba(31,119,180,0.8)'; g = 'rgba(95,190,68,0.8)'; r = 'rgba(220,25,30,0.8)'
b = 'rgba(31,119,180,0.4)'; g = 'rgba(95,190,68,0.4)'; r = 'rgba(220,25,30,0.4)'

'''color end'''


def plot_sankey(scenario, landuse_matrix, tech_wwt, tech_GP1, tech_GP2, tech_GP3):
    # tech_wwt = 'AS'; tech_GP1 = 1; tech_GP2 = 1; tech_GP3 = 1
    # tech_wwt = 'EBPR_StR'; tech_GP1 = 2; tech_GP2 = 2; tech_GP3 = 2
    s1 = ITEEM(landuse_matrix, tech_wwt=tech_wwt, limit_N=10.0, tech_GP1=tech_GP1, tech_GP2=tech_GP2, tech_GP3=tech_GP3)
    output_s1 = s1.get_P_flow()
    value = output_s1[-1]
    P_soil_fertilizer = output_s1[-2]
    # if P_soil_fertilizer > 0:
    #     label = ["Imported corn,0", "Fertilizer,1", "Manure,2", "Wastewater treatment plant,3", 
    #               'Corn biorefineries (CBs),4', "In-stream loads,5", "Local corn,6", 'Products from CBs,7', 
    #               'recovered P,8', 'Local soybean,9', 'Corn silage,10', 'Soil,11', 'Riverine export,12',
    #               'Reservoir trapping,13', 'In-stream storage,14', 'Biomass,15', 'Biosolid,16',
    #               'Soybean biorefinery,17', 'Products from soybean biorefinery,18', 
    #               'Soybean (exported),19', 'Human wastewater,20'
    #               ]
        
    #     # label = ['']
    #     source = [0, 1, 1, 1, 1,  1,  2, 2,  2,  3, 3, 3, 3,  4, 4, 5,  5,  5, 6,  9,  9, 17, 17, 20, 4, 4]
    #     target = [4, 5, 6, 9, 11, 15, 5, 10, 11, 5, 6, 8, 11, 7, 8, 12, 13, 14,4, 17, 19, 18, 3,  3 , 3, 2]
        
    #     # color_node3 = []
    #     # for i in source:
    #     #     color_node3.append(color_node[i])
    #     #     color_link3 = [i.replace('0.8', '0.4') for i in color_node3]
            
    #     b = 'rgba(31,119,180,0.8)'; g = 'rgba(17,100,5,0.8)'; r = 'rgba(20,20,20,0.6)'
    #     color_node = [g, r, r, b, r, b, g, r, r, g, g, r, b, b, b, g, r, r, r, g, b]

    #     b = 'rgba(31,119,180,0.4)'; g = 'rgba(17,100,5,0.4)'; r = 'rgba(20,20,20,0.2)'
    #     color_link = [g, r, r, r, r, r, r, r, r, b, r, b, b,  r, r,  b, b,  b,  g, g,  g,  r, r, b,  r, r]        

    #     fig = go.Figure(go.Sankey(
    #         arrangement = "snap", 
    #         valueformat = ".0f",
            
    #         node = dict(
    #             label = label,
    #             x = [0.0, 0.0, 0.6, 0.5, 0.5, 0.70, 0.35, 1.0, 0.2, 1.0,   1, 1.0, 1.0, 0, 0.35, 1.0, 1.0, 0.0, 0.0],
    #             y = [0.0, 0.9, 0.9, 1.0, 0.5, 0.95, 0.73, 0.5, 0.9, 0.6, 0.5, 1.0, 0.8, 0, 0.95, 0.8, 0.81, 0.8, 0.0],
    #             pad = 10,
    #             line = dict(color = "black", width = 1),
    #             color = color_node
    #             ),  # 10 Pixels
            
    #         link = {
    #                 "source": source,
    #                 "target": target,
    #                 "value" : value,
    #                 'color': color_link #color_link3
    #                 }))

    # elif P_soil_fertilizer < 0:
    #     # label = ["Imported corn", "Fertilizer", "Manure", "Wastewater treatment plant", 
    #     #           'Corn biorefineries (CBs)', "In-stream loads", "Local corn", 
    #     #           'Products from CBs', 'recovered P', 'Soybean', 'Corn silage', 'Soil', 
    #     #           'Riverine export', 'Reservoir trapping', 'In-stream storage', 'Biomass','Biosolid']
        
    #     label = ["Imported corn,0", "Fertilizer,1", "Manure,2", "Wastewater treatment plant,3", 
    #               'Corn biorefineries (CBs),4', "In-stream loads,5", "Local corn,6", 'Products from CBs,7', 
    #               'recovered P,8', 'Local soybean,9', 'Corn silage,10', 'Soil,11', 'Riverine export,12',
    #               'Reservoir trapping,13', 'In-stream storage,14', 'Biomass,15', 'Biosolid,16',
    #               'Soybean biorefinery,17', 'Products from soybean biorefinery,18', 
    #               'Soybean (exported),19', 'Human wastewater,20'
    #               ]
        
    #     label = ['']
    #     source = [0, 1, 1, 1, 1,  2,  2, 3, 3, 3, 3, 4, 4,  5,  5,  5, 6, 11, 11, 9,  9, 17, 17, 20, 4, 4]
    #     target = [4, 5, 6, 9, 15, 5, 10, 5, 6, 8, 16,7, 8, 12, 13, 14, 4, 6,  9, 17, 19, 18, 3,  3 , 3, 2]
    #     # color_node3 = []
    #     # for i in source:
    #     #     color_node3.append(color_node[i])
    #     #     color_link3 = [i.replace('0.8', '0.4') for i in color_node3]
    #     b = 'rgba(31,119,180,0.8)'; g = 'rgba(17,100,5,0.8)'; r = 'rgba(20,20,20,0.6)'
    #     color_node = [g, r, r, b, r, b, g, r, r, g, g, r, b, b, b, g, r, r, r, g, b]
    #     b = 'rgba(31,119,180,0.4)'; g = 'rgba(17,100,5,0.4)'; r = 'rgba(20,20,20,0.2)'
    #     color_link = [g, r, r, r, r, r, r, b, r, b,  b, r, r, b, b, b, g, r, r, g,  g,  r, r, b, r, r]   
        
    #     fig = go.Figure(go.Sankey(
    #         arrangement = "snap",
    #         valueformat = ".0f",
    #         node = dict(
    #             label = label,
    #             x = [0.01, 0.01, 0.6, 0.6,  0.5, 0.65, 0.3,  1.0, 1.0, 0.2, 1.0, 0.01, 1,  1,  1.0, 1.0,  0.3,   1,  1],
    #             y = [0.45, 0.55, 0.9, 0.85, 0.5, 0.95, 0.8, 0.3, 0.5, 1.0, 0.8, 1.0, 0.8,0.9,0.89, 0.8, 0.95, 0.95, 1.0],
    #             pad = 10,# 10 Pixels
    #             line = dict(color = "black", width = 0.5),
    #             color = color_node
    #             ),
            
    #         link = {
    #                 "source": source,
    #                 "target": target,
    #                 "value" : value,
    #                 'color': color_link#color_link3
    #                 }))
    
    
    
    '''Small figure'''
    if P_soil_fertilizer > 0:
        label = ['Imported corn,0', 'Fertilizer,1', 'Manure,2', 'Wastewater treatment plant,3', 
                  'Corn biorefineries (CBs),4', 'In-stream loads,5', 'Local corn,6', 'Products from CBs,7', 
                  'recovered P,8', 'Local soybean,9', 'Corn silage,10', 'Soil,11', 'Riverine export,12',
                  'Reservoir trapping,13', 'In-stream storage,14', 'Biomass,15', 'Biosolid,16',
                  'Soybean biorefinery,17', 'Products from soybean biorefinery,18',
                  'Soybean (exported),19', 'Human wastewater,20']
        
        value = [value[1], value[9], value[15], value[16], 
                 value[17], value[-3], value[-4], value[-2], value[12], value[10]]    
        label = ['']
        source = [1, 3, 5,   5, 5,  20, 17, 4, 3, 3]
        target = [5, 5, 12, 13, 14, 3, 3,  3, 16, 6]
        # value = output_s1[-1]
        
        b = 'rgba(31,119,180,0.8)'; g = 'rgba(17,100,5,0.8)'; r = 'rgba(20,20,20,0.6)'
        color_node = [g, r, r, b, r, b, g, r, r, g, g, r, b, b, b, g, r, r, r, g, b]
        b = 'rgba(31,119,180,0.4)'; g = 'rgba(17,100,5,0.4)'; r = 'rgba(20,20,20,0.2)'
        color_link = [r, b, b, b, b, b, r, r, b, b]    

        fig = go.Figure(go.Sankey(
            arrangement = "snap", 
            valueformat = ".0f",
            
            node = dict(
                label = label,
                # x = [0.0, 0.0, 0.6, 0.5, 0.5, 0.70, 0.35, 1.0, 0.2, 1.0,   1, 1.0, 1.0, 0, 0.35, 1.0, 1.0, 0.0, 0.0],
                # y = [0.0, 0.9, 0.9, 1.0, 0.5, 0.95, 0.73, 0.5, 0.9, 0.6, 0.5, 1.0, 0.8, 0, 0.95, 0.8, 0.81, 0.8, 0.0],
                pad = 10,
                line = dict(color = "black", width = 1),
                color = color_node
                ),  # 10 Pixels
            
            link = {
                    "source": source,
                    "target": target,
                    "value" : value,
                    'color': color_link #color_link3
                    }))

    elif P_soil_fertilizer < 0:

        label = ["Imported corn,0", "Fertilizer,1", "Manure,2", "Wastewater treatment plant,3", 
                  'Corn biorefineries (CBs),4', "In-stream loads,5", "Local corn,6", 'Products from CBs,7', 
                  'recovered P,8', 'Local soybean,9', 'Corn silage,10', 'Soil,11', 'Riverine export,12',
                  'Reservoir trapping,13', 'In-stream storage,14', 'Biomass,15', 'Biosolid,16',
                  'Soybean biorefinery,17', 'Products from soybean biorefinery,18', 
                  'Soybean (exported),19', 'Human wastewater,20'
                  ]
        
        value = [value[1], value[7], value[13], value[14], value[15], value[9],
                 value[-3], value[-4], value[-2], value[10], value[8]
                 ]   
        label = ['']
        source = [1, 3, 5,   5, 5,  3, 20, 17, 4, 3,  3]
        target = [5, 5, 12, 13, 14, 8, 3,  3,  3, 16, 6]

        b = 'rgba(31,119,180,0.8)'; g = 'rgba(17,100,5,0.8)'; r = 'rgba(20,20,20,0.6)'
        color_node = [g, r, r, b, r, b, g, r, r, g, g, r, b, b, b, g, r, r, r, g, b]
        b = 'rgba(31,119,180,0.4)'; g = 'rgba(17,100,5,0.4)'; r = 'rgba(20,20,20,0.2)'
        color_link = [r, b, b, b, b, b, b, r, r, b, b]   
        fig = go.Figure(go.Sankey(
            arrangement = "snap",
            valueformat = ".0f",
            node = dict(
                label = label,
                # x = [0,0,0,0,0.5,0.5],
                # y = [0,0,0,0,1.0,1.0] ,
                pad = 10,# 10 Pixels
                line = dict(color = "black", width = 0.5),
                color = color_node
                ),
            
            link = {
                    "source": source,
                    "target": target,
                    "value" : value,
                    'color': color_link#color_link3
                    }))
    
    
    
    fig.update_layout(font_size=15)
    fig.show()
    plot(fig)
    #save a figure of 300dpi, with 1.5 inches, and  height 0.75inches
    pio.write_image(fig, scenario + '_Oct21.jpg', width=3.0*300, height=3.0*300, scale=1)
    pio.write_image(fig, scenario + '_Oct21.pdf', width=3.0*300, height=3.0*300, scale=1)

    return value, P_soil_fertilizer

# landuse_matrix = np.zeros((45,62))
# landuse_matrix[:, 1] = 1
# landuse_matrix[:,55] = 0.5
# value_baseline = plot_sankey('baseline_Oct21_2021', landuse_matrix, 'AS', 1, 1,1)

landuse_matrix = np.zeros((45,62))
landuse_matrix[:,47] = 1
value_BMP47_EBPR_StR = plot_sankey('BMP47_EBPR_StR',landuse_matrix,'EBPR_StR',2,2,2)