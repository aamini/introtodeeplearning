
# MIT 6.S191: Introduction to Deep Learning 
# with Docker Toolbox for Windows


## Step 01 - Download Docker Toolbox

[Click here to download Docker Toolbox](https://download.docker.com/win/stable/DockerToolbox.exe)

DockerToolbox.exe will install

* Docker Client for Windows
* Docker Toolbox management tool and ISO
* Oracle VM VirtualBox
* Git MSYS-git UNIX tools


[**Click here if you want step by step Install Docker Toolbox on Windows**](https://docs.docker.com/toolbox/toolbox_install_windows/)


## Step 02 - Note

Because the Docker Engine daemon uses Linux-specific kernel features, you can't run Docker Engine natively on Windows. Instead, you must use the Docker Machine command, docker-machine , to create and attach to a small Linux VM on your machine. This VM hosts Docker Engine for you on your Windows system.

To run Docker Linux VM on windows there is two ways

1. Virtualbox:
a free and open-source hypervisor for x86 computers currently being developed by Oracle Corporation

https://download.docker.com/win/stable/DockerToolbox.exe

2. Hyper-V :
virtualization technology built into Windows 10

https://download.docker.com/win/stable/InstallDocker.msi

Where will be useing **Virtualbox**

## Step 03 - cloneing the labs

on the keybord press

    Windows Key + R

wait...

on the Run window text-Input write 

    cmd

click "OK"

on the CMD window write 

    set projectPath=D:\Experiments\E.introtodeeplearning_labs3
    mkdir %projectPath%
    cd /d %projectPath%
    git clone https://github.com/aamini/introtodeeplearning_labs.git .
    exit

## Step 04 - Note

chage the projectPath to where you want your

## Step 05 - opening docker
    
On the keybord press

    Windows Key + S
    
now seach an open 

    Docker Quickstart Terminal

wait untill to see

                            ##         .
                      ## ## ##        ==
                   ## ## ## ## ##    ===
               /"""""""""""""""""\___/ ===
          ~~~ {~~ ~~~~ ~~~ ~~~~ ~~~ ~ /  ===- ~~~
               \______ o           __/
                 \    \         __/
                  \____\_______/

    docker is configured to use the default machine with IP 192.168.99.100
    For help getting started, check out the docs at https://docs.docker.com


now close the window

## Step 06 - mounting you project

On the keybord press

    Windows Key + R

wait

On the Run window text-Input write 

  cmd

On the CMD window write 

        set projectPath=D:\Experiments\E.introtodeeplearning_labs3

        set VBoxPath=C:\Program Files\Oracle\VirtualBox\

        Set PATH=%VBoxPath%;%PATH%

        docker-machine stop

        VBoxManage sharedfolder ^
        remove default ^
        --name "introtodeeplearning_labs"

        vboxmanage sharedfolder ^
        add default ^
        --name "introtodeeplearning_labs" ^
        --hostpath %projectPath% ^
        --automount
        
        docker-machine start  
    
## Step 07 - Note

chage the projectPath to where you want your

## Step 08 - runing the lab docker image

On the keybord press

    Windows Key + S
    
now seach an open 

    Docker Quickstart Terminal

wait untill to see

                            ##         .
                      ## ## ##        ==
                   ## ## ## ## ##    ===
               /"""""""""""""""""\___/ ===
          ~~~ {~~ ~~~~ ~~~ ~~~~ ~~~ ~ /  ===- ~~~
               \______ o           __/
                 \    \         __/
                  \____\_______/

    docker is configured to use the default machine with IP 192.168.99.100
    For help getting started, check out the docs at https://docs.docker.com

On the Docker Quickstart Terminal window write 

        docker run \
        -p 8888:8888 \
        -p 6006:6006 \
        -v /introtodeeplearning_labs:/notebooks/introtodeeplearning_labs \
        mit6s191/iap2018:labs


wait untill you see:

        Copy/paste this URL into your browser when you connect for the first time,
        to login with a token:
        http://localhost:8888/?token=b1c364678b447155d9a40a3da9cc30f7e837276e2e4d9066



## Step 09 - opening jupyter notebook

now open you brwoser  

    http://localhost:8888/?token=b1c364678b447155d9a40a3da9cc30f7e837276e2e4d9066
  
replacing localhos with 192.168.99.100

    http://192.168.99.100:8888/?token=b1c364678b447155d9a40a3da9cc30f7e837276e2e4d9066


## Step 10 - Running the labs

Now, to run the labs, after opening the Jupyter notebook on

`localhost:8888` and enter the `lab1` or `lab2` directory.

Go through the notebooks and fill in the `#TODO` cells 

to get the code to compile for yourself!


## Step 11 - Note

After you , but for me docker Toolbox has 
a lot of thing that can go wrong so the 
1st thing you can do is deleteing Docker 
Toolbox default vbox image

On the keybord press

    Windows Key + R

wait

On the Run window text-Input write 

    cmd

On the CMD window write 
    
    set VBoxPath=C:\Program Files\Oracle\VirtualBox\
    Set PATH=%VBoxPath%;%PATH%
    vboxmanage unregistervm default --delete

now reapeat **Step 05** to **Step 10**
