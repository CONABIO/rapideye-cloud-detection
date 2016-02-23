# Ubuntu 14.04 Trusty Tahyr
FROM geodata/gdal:1.11.2

MAINTAINER Amaury Gutierrez <amaury.gtz@gmail.com>
# change user from none to root
USER root
# install python dependencies
RUN apt-get update -y && \
	apt-get install -y \
	python-skimage 
RUN pip install ephem

