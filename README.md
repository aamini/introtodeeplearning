# MIT 6.S191: Introduction to Deep Learning

## Installation:
To run these labs, first you must install the class docker container, which is available for free on the DockerHub. This can be done by first installing Docker: [Ubuntu](https://www.docker.com/docker-ubuntu), [Mac OSX](https://www.docker.com/docker-mac), [Windows](https://www.docker.com/docker-windows)

For additional help to install Docker on Windows please see: [this link](WindowsDocker.md) -- thanks to Elrashid for putting it together! 

## Starting the enviornment
Once you donwload docker all you need to do is run the container to start! This can be done by running the following command in your command terminal:
```
docker run -p 8888:8888 -p 6006:6006 -v /path/to/introtodeeplearning_labs:/notebooks/introtodeeplearning_labs mit6s191/iap2018:labs
```
Make sure you replace `/path/to/introtodeeplearning_labs` with the correct path to this github repo.

## Running the labs
Now, to run the labs, open the Jupyter notebook on `localhost:8888` and enter the `lab1` or `lab2` directory. Go through the notebooks and fill in the `#TODO` cells to get the code to compile for yourself!

