from os import listdir
from os.path import isfile, join
from xml.dom.minidom import parse
from skimage import color
from scipy import ndimage
from scipy import signal
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as pylab
import sys
import datetime
import re
import gdal
import numpy
import os.path
import osr
import cloud.util as util

FMASK_LAND = 0
FMASK_WATER = 1
FMASK_SNOW = 3
FMASK_OUTSIDE = 255
FMASK_CLOUD_SHADOW = 2
FMASK_CLOUD = 4

def main(directory):
    '''
    Main method.
    '''
    tif_file_regex = re.compile(".*[0-9]{6}.tif$")
    metadata_file_regex = re.compile(".*[0-9]{6}_metadata.xml$")
    
    for f in listdir(directory):
      if isfile(join(directory, f)): 
      	if metadata_file_regex.match(f):
        	metadata = join(directory, f)
        if tif_file_regex.match(f):
        	image_path = join(directory, f)
    
    metadata_xml = parse(metadata)
    solar_zenith = util.get_float_metadata(metadata_xml, 'opt:illuminationElevationAngle')
    # We parse the aquisition date into a format that can be later used to calculate distance to the sun.
    aquisition_date = datetime.datetime.strptime(util.get_metadata(metadata_xml, 'eop:acquisitionDate'), "%Y-%m-%dT%H:%M:%S.%fZ")
    solar_azimuth =  util.get_float_metadata(metadata_xml, 'opt:illuminationAzimuthAngle')


    sun_earth_distance = calculate_distance_sun_earth(aquisition_date)


    print sun_earth_distance
    print aquisition_date
    print solar_zenith
    print solar_azimuth

    image_dictionary = util.get_data_from_image(image_path)
    original_info = image_dictionary['array']

    # Obtain the radiance from the raw data array.
    radiance = calculate_rad_rapideye(original_info)
    # With the radiance, we can calculate the top of atmosphere.
    
    top_of_atmosphere_data = calculate_toa_rapideye(radiance, sun_earth_distance, solar_zenith)


    # Create file routine.
    base, ext = os.path.splitext(image_path)

    
    # We write the top of atmosphere to a file.
    toa_output_file = '%s_toa.tif' % base
    util.write_array_to_tiff(top_of_atmosphere_data, toa_output_file, top_of_atmosphere_data.shape[0], image_dictionary)


    # We calculate the cloud mask, and write it to a file.
    cloud_output_file = '%s_cloud.tif' % base
    clouds = base_masking_rapideye(top_of_atmosphere_data, base, solar_zenith, solar_azimuth, image_dictionary['geotransform'])
    util.write_array_to_tiff(clouds, cloud_output_file, 1, image_dictionary)




def calculate_distance_sun_earth(datestr):
    '''
    Calculates distance between sun and earth in astronomical unints for a given
    date. Date needs to be a string using format YYYY-MM-DD or datetime object
    from metadata.
    '''
    import ephem
    sun = ephem.Sun()
    if isinstance(datestr, str):
        sun.compute(datetime.datetime.strptime(datestr, '%Y-%m-%d').date())
    elif isinstance(datestr, datetime.datetime ):
        sun.compute(datestr)
    sun_distance = sun.earth_distance  # needs to be between 0.9832898912 AU and 1.0167103335 AU
    return sun_distance
def calculate_rad_rapideye(data, radiometricScaleFactor=0.009999999776482582, radiometricOffsetValue=0.0):
    '''
    Convert digital number into radiance according to rapideye documentation. Returns 
    sensor radiance of that pixel in watts per steradian per square meter.
    '''
    rad = data * radiometricScaleFactor + radiometricOffsetValue
    rad[rad == radiometricOffsetValue] = 0
    return rad
def calculate_toa_rapideye(rad, sun_distance, sun_elevation):
    '''
    Calculates top of atmosphere from radiance according to RE documentation
    needs sun-earth distance and sun elevation.
    '''
    print 'sun_elevation: %s' % sun_elevation
    BANDS = 5
    solar_zenith = 90 - sun_elevation
    EAI = [1997.8, 1863.5, 1560.4, 1395.0, 1124.4]  # Exo-Atmospheric Irradiance in 
    toa = rad
    for i in range(BANDS):
        toa[i, :, :] = rad[i, :, :] * (numpy.pi * (sun_distance * sun_distance)) / (EAI[i] * numpy.cos(numpy.pi * solar_zenith / 180)) 
    return toa
def base_top_of_atmosphere_rapideye(sensor_metadata, array):
    from madmex.mapper.sensor import rapideye
    solar_zenith = sensor_metadata(rapideye.SOLAR_ZENITH)
    data_acquisition_date = sensor_metadata(rapideye.ACQUISITION_DATE)
    sun_earth_distance = calculate_distance_sun_earth(data_acquisition_date)
    top_of_atmosphere_data = calculate_toa_rapideye(calculate_rad_rapideye(array), sun_earth_distance, solar_zenith)
    return top_of_atmosphere_data

def base_masking_rapideye(top_of_atmosphere_data, basename, solar_zenith, solar_azimuth, geotransform):
    #from madmex.mapper.sensor import rapideye
    #from madmex.mapper.data import raster
    #solar_zenith = fun_get_attr_sensor_metadata(rapideye.SOLAR_ZENITH)
    #solar_azimuth = fun_get_attr_sensor_metadata(rapideye.SOLAR_AZIMUTH)
    #geotransform = fun_get_attr_raster_metadata(raster.GEOTRANSFORM)
    make_plot = True
    cloud_mask = convert_to_fmask(extract_extremes(top_of_atmosphere_data, basename, make_plot))
    clouds = numpy.where(cloud_mask==FMASK_CLOUD, 1, 0)
    shadows= numpy.where(cloud_mask==FMASK_CLOUD_SHADOW, 1, 0)
    resolution = geotransform[1]
    shadow_mask = calculate_cloud_shadow(clouds, shadows, solar_zenith, solar_azimuth, resolution) * FMASK_CLOUD_SHADOW
    water_mask = convert_water_to_fmask(calculate_water(top_of_atmosphere_data))
    return combine_mask(cloud_mask, shadow_mask, water_mask)

def extract_extremes(data, basename, make_plot, steps=1000):
    #TODO: the two for and the last one takes too long to finish
    CENTER = 50
    xtiles = 5000 / steps
    ytiles = 5000 / steps
    counter = 0
    b = 0
    global_quant = list()
    for ycount in range(0, 5000, steps):
        for xcount in range(0, 5000, steps):
            subset = data[:, ycount:ycount + steps, xcount:xcount + steps]
            z, y, x = subset.shape
            rgb = numpy.zeros((y, x, z))
            for i in range(0, z):
                rgb[:, :, i] = subset[i, :, :] / numpy.max(data[i, :, :])
            col_space1 = color.rgb2lab(rgb[:, :, 0:3])
            quant = calculate_quantiles(col_space1[ :, :, 0])
            if len(quant) > 0:
                points = calculate_breaking_points(quant)
                break_points = numpy.array(calculate_continuity(points.keys()))
            else:
                break_points = []
            subset = data[b, ycount:ycount + steps, xcount:xcount + steps]
            if make_plot:
                pylab.subplot(xtiles, ytiles, counter + 1)
                fig = pylab.plot(quant, c="k", alpha=0.5)
            if len(break_points) > 1:
                if make_plot:
                    for p in points.keys():
                        pylab.plot(p, numpy.array(quant)[p], 'r.')
                if len(break_points[break_points < CENTER]) > 0:
                    min_bp = max(break_points[break_points < CENTER])
                if len(break_points[break_points > CENTER]) > 0:
                    max_bp = min(break_points[break_points > CENTER])
                if numpy.min(subset) != 0.0 or numpy.max(subset) != 0.0:
                    print "%d, %3.2f, %3.2f, %3.2f * %d, %3.2f, %3.2f, %3.2f" % (min_bp, quant[min_bp], numpy.min(subset), quant[0] / numpy.min(subset), max_bp, quant[99] , numpy.max(subset), quant[99] / numpy.max(subset))
                for i, q in enumerate(range(max_bp, 100, 1)):
                    modelled = f_lin(q, points[max_bp]["slope"], points[max_bp]["offset"])
                    diff = abs(quant[q] - modelled)
                    global_quant.append((q, quant[q], diff))
                for i, q in enumerate(range(0, min_bp,)):
                    modelled = f_lin(q, points[max_bp]["slope"], points[max_bp]["offset"])
                    diff = abs(quant[q] - modelled)
                    global_quant.append((q, quant[q], diff))
            counter += 1
            if make_plot:
                fig = pylab.gcf()
                fig.set_size_inches(18.5, 10.5)
                name = "%s_local.png" % basename
                fig.savefig(name, bbox_inches='tight', dpi=150)
    z, y, x = data.shape
    rgb = numpy.zeros((y, x, z))
    for i in range(0, z):
        rgb[:, :, i] = data[i, :, :] / numpy.max(data[i, :, :])
    col_space1 = color.rgb2lab(rgb[:, :, 0:3])
    subset_result = numpy.zeros((3, y, x), dtype=numpy.float)
    total_q = len(global_quant)
    for counter, item in enumerate(global_quant):
        print "%d: %d, %s" % ( counter, total_q, str(item))
        q, value, diff = item
        MAX_ERROR = 10
        if diff > MAX_ERROR:
            if q < 50.0:
                subset_result[0, :, :] = numpy.where(col_space1[ :, :, 0] < value, 1 + col_space1[ :, :, 0] - diff, subset_result[0, :, :])
                subset_result[1, :, :] = numpy.where(col_space1[ :, :, 0] < value, subset_result[1, :, :] + diff, subset_result[1, :, :])
                subset_result[2, :, :] = numpy.where(col_space1[ :, :, 2] < value, subset_result[2, :, :] - 1, subset_result[2, :, :])
            else:                        
                subset_result[0, :, :] = numpy.where(col_space1[ :, :, 0] > value, 100. + col_space1[ :, :, 0] - diff, subset_result[0, :, :])
                subset_result[1, :, :] = numpy.where(col_space1[ :, :, 0] > value, subset_result[1, :, :] + diff, subset_result[1, :, :])
                subset_result[2, :, :] = numpy.where(col_space1[ :, :, 2] > value, subset_result[2, :, :] + 1, subset_result[2, :, :])
    return subset_result

def calculate_cloud_shadow(clouds, shadows, solar_zenith, solar_azimuth, resolution):
    '''
    This method iterates over a list of different cloud heights, and calculates
    the shadow that the clouds in the given mask project. This projections are
    then intersected with the shadow mask that we already have.
    '''
    cloud_row_column = numpy.column_stack(numpy.where(clouds == 1))
    cloud_heights = numpy.arange(1000, 3100, 100)
    cloud_mask_shape = clouds.shape
    clouds_projection = numpy.zeros(cloud_mask_shape)  
    for cloud_height in cloud_heights:    
        distance = cloud_height / resolution * numpy.tan(numpy.deg2rad(90 - solar_zenith))
        y_difference = distance * numpy.sin(numpy.deg2rad(360 - solar_azimuth))
        x_difference = distance * numpy.cos(numpy.deg2rad(360 - solar_azimuth))     
        if solar_azimuth < 180:
            rows = cloud_row_column[:, 0] - y_difference #/ 5
            cols = cloud_row_column[:, 1] - x_difference #/ 5
        else:
            rows = cloud_row_column[:, 0] + y_difference #/ 5
            cols = cloud_row_column[:, 1] + x_difference #/ 5
        rows = rows.astype(numpy.int)
        cols = cols.astype(numpy.int)
        numpy.putmask(rows, rows < 0, 0)
        numpy.putmask(cols, cols < 0, 0)
        numpy.putmask(rows, rows >= cloud_mask_shape[0] - 1, cloud_mask_shape[0]  - 1)
        numpy.putmask(cols, cols >= cloud_mask_shape[1]  - 1, cloud_mask_shape[1]  - 1)
        clouds_projection[rows, cols] = 1  
    in_between = shadows * clouds_projection
    return in_between
def convert_to_fmask(raster):
    from numpy.core.numerictypes import byte
    z, y, x = raster.shape
    fmask = numpy.zeros((y, x)).astype(byte)  # FMASK_LAND
    fmask = numpy.where (raster[0, :, :] > 100, FMASK_CLOUD, fmask)
    fmask = numpy.where (raster[0, :, :] < 100, FMASK_CLOUD_SHADOW, fmask)
    fmask = numpy.where (raster[0, :, :] == 0, FMASK_LAND, fmask)
    fmask = numpy.where (raster[0, :, :] == -999.0, FMASK_OUTSIDE, fmask)
    return fmask
def convert_water_to_fmask(raster):
    from numpy.core.numerictypes import byte
    y, x = raster.shape
    fmask = numpy.zeros((y, x)).astype(byte)  # FMASK_LAND
    fmask = numpy.where (raster > 1, FMASK_WATER, fmask)
    fmask = numpy.where (raster <= 1, FMASK_LAND, fmask)
    fmask = numpy.where (raster == -999.0, FMASK_OUTSIDE, fmask)
    return fmask
def combine_mask(cloudmask, shadowmask, watermask):
    watermask = numpy.where(watermask == FMASK_CLOUD_SHADOW, FMASK_CLOUD_SHADOW, watermask)
    watermask = numpy.where(cloudmask == FMASK_CLOUD, FMASK_CLOUD, watermask)
    watermask = numpy.where(shadowmask == FMASK_CLOUD_SHADOW, FMASK_CLOUD_SHADOW, watermask)
    return watermask
def calculate_quantiles(band, NA=0):
    quant = list()
    na_free_data = band[band > NA]
    if len(na_free_data) > 100:
        for i in range(0, 100, 1):
            p = numpy.percentile(na_free_data, i)
            quant.append(p)
    else:
        quant = []
    return numpy.array(quant)
def calculate_breaking_points(quant_list):
    MAX_ERROR = 5
    RANGE = 5
    CENTER = 50
    BRAKE_POINTS = dict()
    # right to left window
    for x_iter in range(100, CENTER, -1):
        x_proj = range(x_iter - RANGE, x_iter)
        # pylab.plot(x_proj, quant[x-RANGE:x], 'k',alpha=0.5)
        y_subset = quant_list[x_iter - RANGE:x_iter]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_proj, y_subset) 
        g, l = calculate_error(slope, intercept, x_proj, y_subset)
        # print x-RANGE,x, slope, intercept, y[0], f(x, slope, intercept), l,r
        if l > MAX_ERROR:
            BRAKE_POINTS[x_iter - RANGE / 2] = {"error":l, "slope":slope, "offset":intercept}
    # left to right window
    for x_iter in range(0, CENTER, 1):
        x_proj = range(x_iter, x_iter + RANGE)
        # pylab.plot(x_proj, quant_list[x_iter:x_iter + RANGE], 'k', alpha=0.3)
        y_subset = quant_list[x_iter:x_iter + RANGE]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_proj, y_subset) 
        g, l = calculate_error(slope, intercept, x_proj, y_subset)
        # print x,x+RANGE, slope, intercept, y[0], f(x, slope, intercept), l,r
        if l > MAX_ERROR:
            if ((x_iter - RANGE + x_iter) / 2) not in BRAKE_POINTS.keys():
                BRAKE_POINTS[x_iter + (RANGE / 2)] = {"error":l, "slope":slope, "offset":intercept}
        # pylab.plot([x_iter, x_iter + RANGE, ], [ f(x_iter, slope, intercept), f(x_iter + RANGE, slope, intercept)], "b", alpha=0.3)
    return BRAKE_POINTS
def calculate_error(slope, offset, x_val, y_val):
    CENTER = 50.0000001
    err_sum_total = 0.0
    err_sum_local = 0.0
    for i, x in enumerate(x_val):
            err_sum_local += ((f_lin(x, slope, offset) - y_val[i]) ** 2) * (CENTER - x) ** 2
            err_sum_total += ((f_lin(x, slope, offset) - y_val[i]) ** 2) * (CENTER - x) ** 2
    return err_sum_total, err_sum_local
def f_lin(x, scale, off):
    return x * scale + off
def calculate_continuity(point_list):
    breaks = list()
    point_list.sort()
    for i in range(1, len(point_list)):
        if point_list[i] - point_list[i - 1] != 1:
            breaks.append(point_list[i - 1])
            breaks.append(point_list[i])    
    return breaks
def calculate_water(data):
    z,y,x = data.shape
    pot_water = numpy.zeros((y,x))
    for b in range(z-1,1,-1):
        pot_water[:,:] = numpy.where(data[b,:,:]<data[b-1,:,:],pot_water+b*1,pot_water)
    for b in range(z-1,1,-1):
        pot_water[:,:] = numpy.where(data[b,:,:]>data[b-1,:,:],pot_water-b*1,pot_water)
    # NDWI bands
    b=4
    pot_water[:,:] = numpy.where(data[b,:,:]>data[b-3,:,:],pot_water-1,pot_water) 
    #NIR-RE diff to  R-RE aka sun glid
    dark_nir = numpy.where(data[b,:,:]<0.12,data, data*0 )
    pot_water[:,:] = numpy.where(numpy.abs(dark_nir[b,:,:]-dark_nir[b-1,:,:]) < abs(dark_nir[b-1,:,:]-dark_nir[b-2,:,:]),pot_water+1.5,pot_water-1.5) 
    pot_water = signal.medfilt2d(pot_water,kernel_size=5)
    pot_water = numpy.where(data[0,:,:]==0, -999, pot_water)
    return pot_water
if __name__ == '__main__':
  directory = sys.argv[1]
  main(directory)