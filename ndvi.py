import os
from osgeo import gdal



output_folder = r'input_gray\ndvi'

# 红外波段图像文件
red_filename =  r'input_gray\ori/L4.png'


# 绿外波段图像文件
green_filename =  r'input_gray\ori\L3.png'


# 近红外波段图像文件
nir_filename =  r'input_gray\ori/L5.png'


# band6  TM5
band6_filename = r'input_gray\ori\L6.png'

output_filename = os.path.join(output_folder, r'Ndvi_plant.TIF')
output_filename_water = os.path.join(output_folder, r'Ndvi_water.TIF')
output_filename_hourse = os.path.join(output_folder, r'Ndvi_house.TIF')



# 打开 两个波段图像
red_dataset = gdal.Open(red_filename, gdal.GA_ReadOnly)
green_dataset = gdal.Open(green_filename, gdal.GA_ReadOnly)
nir_dataset = gdal.Open(nir_filename, gdal.GA_ReadOnly)
band6_dataset = gdal.Open(band6_filename, gdal.GA_ReadOnly)

width = red_dataset.RasterXSize
height = red_dataset.RasterYSize

driver = gdal.GetDriverByName('GTiff')



# 获取红外波段
band_red = red_dataset.GetRasterBand(1)
data_red = band_red.ReadAsArray()
data_red = data_red.astype(float)

# 获取绿红外波段
band_green = green_dataset.GetRasterBand(1)
data_green = band_green.ReadAsArray()
data_green = data_green.astype(float)

# 获取近似红外波段
band_nir = nir_dataset.GetRasterBand(1)
data_nir = band_nir.ReadAsArray()
data_nir = data_nir.astype(float)


# 获取band6波段
band_6 = band6_dataset.GetRasterBand(1)
data_band6 = band_6.ReadAsArray()
data_band6 = data_band6.astype(float)

# 计算归一化植被指数ndvi
data_ndvi = (data_nir - data_red) / (data_nir + data_red)
# 计算归一化水体指数ndvi
data_water_ndvi = (data_green - data_nir) / (data_green + data_nir)
# 计算归一化建筑知识指数ndbi   (B6-B5)/(B6+B5)
data_hourse_ndvi = (data_band6 - data_nir) / (data_band6 + data_nir)

# 获取输出图像波段并写入ndvi数据，然后保存
# 新建输出图像
output_dataset = driver.Create(output_filename, width, height, 1, gdal.GDT_Float32)
output_band = output_dataset.GetRasterBand(1)
output_band.WriteArray(data_ndvi)
output_band.FlushCache()
output_dataset.SetGeoTransform(red_dataset.GetGeoTransform())
output_dataset.SetProjection(red_dataset.GetProjection())

dataset = None
output_dataset = None


# 获取输出图像波段并写入水体指数ndvi数据，然后保存
# 新建输出图像
output_dataset2 = driver.Create(output_filename_water, width, height, 1, gdal.GDT_Float32)
output_band = output_dataset2.GetRasterBand(1)
output_band.WriteArray(data_water_ndvi)
output_band.FlushCache()
output_dataset2.SetGeoTransform(red_dataset.GetGeoTransform())
output_dataset2.SetProjection(red_dataset.GetProjection())

dataset = None
output_dataset2 = None


# 获取输出图像波段并写入建筑指数ndvi数据，然后保存
# 新建输出图像
output_dataset3 = driver.Create(output_filename_hourse, width, height, 1, gdal.GDT_Float32)
output_band = output_dataset3.GetRasterBand(1)
output_band.WriteArray(data_hourse_ndvi)
output_band.FlushCache()
output_dataset3.SetGeoTransform(red_dataset.GetGeoTransform())
output_dataset3.SetProjection(red_dataset.GetProjection())

dataset = None
output_dataset3 = None
