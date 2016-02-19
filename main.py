from os import listdir
from os.path import isfile, join
from xml.dom.minidom import parse
import sys
import datetime

def main(directory):
	'''
	Main method.
	'''

	for f in listdir(directory):
		if isfile(join(directory, f)) and f.endswith('xml'):
			metadata = join(directory, f)

	metadata_xml = parse(metadata)

	solar_zenit_element =  metadata_xml.getElementsByTagName('opt:illuminationElevationAngle')
	solar_zenit = get_text(solar_zenit_element[0].childNodes)

	aquisition_date_element =  metadata_xml.getElementsByTagName('eop:acquisitionDate')
	aquisition_date = datetime.datetime.strptime(get_text(aquisition_date_element[0].childNodes), "%Y-%m-%dT%H:%M:%S.%fZ")
	sun_earth_distance = calculate_distance_sun_earth(aquisition_date)


	print sun_earth_distance
	print aquisition_date
	print solar_zenit


def get_text(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)

def calculate_distance_sun_earth(datestr):
    '''
    Calculates distance between sun and earth in astronomical unints for a given
    date. Date needs to be a string using format YYYY-MM-DD or datetime object
    from metadata.
    '''
    import ephem
    sun = ephem.Sun()  # @UndefinedVariable
    if isinstance(datestr, str):
        sun.compute(datetime.datetime.strptime(datestr, '%Y-%m-%d').date())
    elif isinstance(datestr, datetime.datetime ):
        sun.compute(datestr)
    sun_distance = sun.earth_distance  # needs to be between 0.9832898912 AU and 1.0167103335 AU
    return sun_distance

def base_top_of_atmosphere_rapideye(sensor_metadata, array):
    from madmex.mapper.sensor import rapideye
    solar_zenith = sensor_metadata(rapideye.SOLAR_ZENITH)
    data_acquisition_date = sensor_metadata(rapideye.ACQUISITION_DATE)
    sun_earth_distance = calculate_distance_sun_earth(data_acquisition_date)
    top_of_atmosphere_data = calculate_toa_rapideye(calculate_rad_rapideye(array), sun_earth_distance, solar_zenith)
    return top_of_atmosphere_data

if __name__ == '__main__':
	directory = sys.argv[1]
	main(directory)