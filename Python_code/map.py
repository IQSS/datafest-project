## DAY 2: Maps

import descartes
import geopandas
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from bokeh.plotting import figure, save, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, LogColorMapper, ColorBar
from bokeh.palettes import Viridis256 as palette


def getPolyCoords(row, geom, coord_type):
    """Returns the coordinates ('x|y') of edges/vertices of a Polygon/others"""

    # Parse the geometries and grab the coordinate
    geometry = row[geom]
    #print(geometry.type)

    if geometry.type=='Polygon':
        if coord_type == 'x':
            # Get the x coordinates of the exterior
            # Interior is more complex: xxx.interiors[0].coords.xy[0]
            return list( geometry.exterior.coords.xy[0] )
        elif coord_type == 'y':
            # Get the y coordinates of the exterior
            return list( geometry.exterior.coords.xy[1] )

    if geometry.type in ['Point', 'LineString']:
        if coord_type == 'x':
            return list( geometry.xy[0] )
        elif coord_type == 'y':
            return list( geometry.xy[1] )

    if geometry.type=='MultiLineString':
        all_xy = []
        for ea in geometry:
            if coord_type == 'x':
                all_xy.extend(list( ea.xy[0] ))
            elif coord_type == 'y':
                all_xy.extend(list( ea.xy[1] ))
        return all_xy

    if geometry.type=='MultiPolygon':
        all_xy = []
        for ea in geometry:
            if coord_type == 'x':
                all_xy.extend(list( ea.exterior.coords.xy[0] ))
            elif coord_type == 'y':
                all_xy.extend(list( ea.exterior.coords.xy[1] ))
        return all_xy

    else:
        # Finally, return empty list for unknown geometries
        return []

# find state-level shapefile here: ftp://ftp2.census.gov/geo/tiger/TIGER2019/STATE/ by unzip the folder
# download state-level census geographies
us_state_geo = geopandas.read_file('D:\\DataFest_2021\\tl_2019_us_state.shp')
# rename `NAME` variable to `state`
us_state_geo = us_state_geo.rename(columns={'NAME':'state'})
# clean
us_state_geo[["GEOID"]] = us_state_geo[["GEOID"]].apply(pd.to_numeric)
us_state_geo = us_state_geo[(us_state_geo.GEOID < 60) & (us_state_geo.state != "Alaska") & (us_state_geo.state != "Hawaii")]
print(us_state_geo.head(60))
#us_state_geo = us_state_geo.to_crs("EPSG:3395")

# use previously cleaned dataset
US_states_cases_week = pd.read_csv("D:\\DataFest_2021\\PythonApplication1\\PythonApplication1\\US_states_cases_week.csv")
# merge weekly COVID-19 cases with spatial data
US_states_cases_week[["GEOID"]] = US_states_cases_week[["GEOID"]].apply(pd.to_numeric)
data_frames = [us_state_geo, US_states_cases_week]
US_cases_long_week_spatial = reduce(lambda  left,right: pd.merge(left,right,on=['GEOID', 'state'], how='left'), data_frames)
# filter out some states and subset data for only latest week
US_cases_long_week_spatial = US_cases_long_week_spatial[(US_cases_long_week_spatial.GEOID < 60) & (US_cases_long_week_spatial.state != "Alaska") & (US_cases_long_week_spatial.state != "Hawaii")]
US_cases_long_week_spatial = US_cases_long_week_spatial[US_cases_long_week_spatial.week_of_year == max(US_cases_long_week_spatial.week_of_year)]

print(US_cases_long_week_spatial.head(60))

# choropleth map starts here

# set the value column that will be visualised
variable = 'cases_rate_100K'
# set the range for the choropleth values
vmin, vmax = 0, 600
# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(15, 15))
# remove the axis
ax.axis('off')
# add a title and annotation
ax.set_title('COVID-19 cases rates for last week', fontdict={'fontsize': '15', 'fontweight' : '3'})
ax.annotate('Data Sources: Harvard Dataverse, 2020; U.S. Census Bureau, 2019', xy=(0.6, .05), xycoords='figure fraction', fontsize=12, color='#555555')

# create an axes on the right side of ax. The width of cax will be 3% of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.05)

# Create colorbar legend
legend = plt.cm.ScalarMappable(cmap='Greens', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# empty array for the data range
legend.set_array([]) # or alternatively sm._A = []. Not sure why this step is necessary, but many recommends it
# add the colorbar to the figure
fig.colorbar(legend, cax=cax)
#fig.colorbar(legend)
# create map
US_cases_long_week_spatial.plot(column=variable, cmap='Greens', linewidth=0.8, ax=ax, cax=cax, edgecolor='1.0')

plt.show()

# interactive map starts here

# get the Polygon x and y coordinates
US_cases_long_week_spatial['x'] = US_cases_long_week_spatial.apply(getPolyCoords, geom='geometry', coord_type='x', axis=1)
US_cases_long_week_spatial['y'] = US_cases_long_week_spatial.apply(getPolyCoords, geom='geometry', coord_type='y', axis=1)
# show only head of x and y columns
print(US_cases_long_week_spatial[['x', 'y']].head(2))
# make a copy, drop the geometry column and create ColumnDataSource
US_cases_long_week_spatial_df = US_cases_long_week_spatial.drop('geometry', axis=1).copy()
gsource = ColumnDataSource(US_cases_long_week_spatial_df)

# create the color mapper
color_mapper = LogColorMapper(palette=palette)
# initialize our figure
p = figure(title="COVID-19 cases rates for last week", plot_width=1200, plot_height=800)
# Plot grid
p.patches('x', 'y', source=gsource, fill_color={'field': 'cases_rate_100K', 'transform': color_mapper}, fill_alpha=1.0, line_color="black", line_width=0.05)
# create a color bar
color_bar = ColorBar(color_mapper=color_mapper, width=16,  location=(0,0))
# add color bar to map
p.add_layout(color_bar, 'right')
# initialize our tool for interactivity to the map
my_hover = HoverTool()
# tell to the HoverTool that what information it should show to us
my_hover.tooltips = [('name', '@state'), ('cases rate', '@cases_rate_100K')]
# add this new tool into our current map
p.add_tools(my_hover)
# save the map
output_file("covid-19_map_hover.html")
save(p)
# show the map
show(p)