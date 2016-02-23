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
       -v <path to rapideye directory>:<path to rapideye directory> \
       -v $(pwd):/data \
       rapideye-cloud/v2 \
       python main.py <path to rapideye directory>
```