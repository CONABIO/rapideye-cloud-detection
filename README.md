# rapideye-cloud-detection
The purpose of this repository is to isolate the cloud detection process for rapideye scenes. In order to do so, we use a in-house approach that consist on detecting anomalies by measuring the expected behavior of the values in a particular band and the ones that are actually obtained.

## Disclosure

This procedure is merely experimental, it is a proposal, and by no means it is a final product. It makes no sense to take it as THE *MADMex cloud masking algorithm*. 

## How to Create the Cloud Mask

First of all we need to create the docker image:

```
docker build -t rapideye-clouds .
```

Once we have the docker image built, we start the docker container attaching the current directory, and the directory containing the rapideye scene:

```
docker run -it \
       -v <path to rapideye directory>:<path to rapideye directory inside docker> \
       -v $(pwd):/data \
       rapideye-clouds \
       python main.py <path to rapideye directory inside docker>
```
## Windows

In Windows, Docker is only allowed to mount files that are found in the directory:

```
C:\\Users
```
So our working directory must be in that directory, for the sake of example, I created a directory called example:

```
C:\\Users\example
```
Recalling that the paths for Windows must be prepend by a "/" we have:

```
docker run -it \
       -v /<path to rapideye directory>:<path to rapideye directory inside docker> \
       -v /$(pwd):/data \
       rapideye-clouds \
       python main.py <path to rapideye directory inside docker>
```
For instance, if our working directory is in "C:\\Users\example\l3a":

```
docker run -it \
       -v /c/Users/example/l3a/:/rapideye/ \
       -v /$(pwd):/data \
       rapideye-clouds \
       python main.py /rapideye/
```

## Remark

It can be the case that the RAM is not enough for this process, if that is the case, settings in VirtualBox must be changed to something around 8192Mb.