from model import *
from decode import *

img = './result.jpg'
img_data, module_size, x_locations_list, y_locations_list = model.getDataArea(img)
x_locations_list = list(range(0, 29 * 16 + 1, 16))
y_locations_list = list(range(0, 29 * 16 + 1, 16))
message = decode(img_data, module_size, x_locations_list, y_locations_list)
print(message)
