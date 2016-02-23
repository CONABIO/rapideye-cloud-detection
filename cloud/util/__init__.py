import gdal
import numpy
import osr


GDAL_DRIVER = 'GTiff'

def get_text(nodelist):
    '''
    This helper method extracts text from a node object in a xml tree.
    '''
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)

def get_float_metadata(tree, tag):
    '''
    Gets and parses the text inside a a node with a particular tag into a float.
    '''
    return float(get_metadata(tree, tag))

def write_array_to_tiff(result_array, output_file_path, n, image_dictionary):
    '''
    This method writes the first n of an array in a file named as the output_file_path
    parameter using the data found in the image_dictionary. 
    '''
    driver = gdal.GetDriverByName(GDAL_DRIVER)
    result_image = driver.Create(output_file_path, image_dictionary['width'], image_dictionary['height'], n, gdal.GDT_Int16)
    
    print 'Number of bands:  %s' % n

    if n > 1:
        for band in range(n):
            result_image.GetRasterBand(band + 1).WriteArray(result_array[band, :, :])
    elif n == 1:
        result_image.GetRasterBand(1).WriteArray(result_array)
    else:
        raise Exception('Invalid Value for the number of bands.')

    result_image.SetProjection(str(image_dictionary['projection']))
    result_image.SetGeoTransform(image_dictionary['geotransform'])
    result_image.FlushCache()

def get_data_from_image(image_path):
    '''
    Reads the image and extracts all the information that will be needed later. It
    then closes the image.
    '''
    dataset = gdal.Open(image_path)

    result = {}
    result['array'] = numpy.array(dataset.ReadAsArray())
    result['bands'] = dataset.RasterCount
    result['width'] = dataset.RasterXSize
    result['height'] = dataset.RasterYSize
    result['geotransform'] = dataset.GetGeoTransform()
    result['projection'] = osr.SpatialReference()
    result['projection'].ImportFromWkt(dataset.GetProjectionRef())

    del dataset
    return result

def get_metadata(tree, tag):
    '''
    Gets and parses the text inside a node.
    '''
    elements = tree.getElementsByTagName(tag)
    text = get_text(elements[0].childNodes)
    return text