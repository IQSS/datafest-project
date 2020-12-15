
import geopandas
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

state_file_url = 'https://www2.census.gov/geo/tiger/TIGER2019/STATE/tl_2019_us_state.zip'

with urlopen(state_file_url) as zipresp:
    with ZipFile(BytesIO(zipresp.read())) as zfile:
        zfile.extractall('D:\\DataFest_2021\\temp')

us_state_geo = geopandas.read_file('D:\\DataFest_2021\\temp\\tl_2019_us_state.shp')
print(us_state_geo.head(60))