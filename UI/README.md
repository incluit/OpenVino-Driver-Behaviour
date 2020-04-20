# OpenVINO Driver Management GUI

[![pipeline status](https://gitlab.com/openvinogui/openvino-gui/badges/master/pipeline.svg)](https://gitlab.com/openvinogui/openvino-gui/commits/master)

The following document is a step by step installation and user guide for executing all available use cases related to OpenVINO for Transportation's POC.
First of all, it is mandatory to verify that the following prerequisites are been met in order to successfully install and execute the targeted demo.

## PREREQUISITES

**Minimum Hardware Requisites**


* CPU: Intel Core i7 6th gen
* GPU: Intel HD Graphics 520
* RAM: 8GB
* Integrated or USB Webcam



**Software Requisites**


* 64-bit Ubuntu 16.04 or 18.04
* Git(git) installed
* BOOST Library installed
* “LibAO” and “libsndfile” installed
* OpenVINO toolkit R2 Release installed
* OpenVINO 2019 R3 or higher
* OpenVINO Driver Management cloned and compiled.
* Python 3.6 or greater
* Flask 1.0.2 or greater
* OpenCV-python
* Python3-pip


**OpenVINO for Transportation Use Cases Deployed**

In order to build and compile please check the "Buildind Guide" link for each
 
* DRIVER MANAGEMENT: [TBD]


*Note: Take into account that depending on the OpenVINO's release version the building proceduce might be affected*

* SIMULATION ADD-ONS: [Required Scripts](https://gitlab.com/openvinogui/driver-behaviour-integration/-/wikis/Ros-2-installation) 


## DEMO UI USER GUIDE


1. Open a Linux terminal console

2. Move to the Driver Management Repoitory and open the UI folder:

```
cd ui/
```

3. Install the requirements for the execution:

```
pip3 install -r requirements.txt
```

*Note: If the user haven't installed python-pip3, please run the command: apt-get install -y python3-pip*

4. Execute the following command line:

```
python3 webapp.py
```

5. Open in your preferred browser: `http://127.0.0.1:5000/`


6. Demo UI will be displayed.


*Note: If the user closes the terminal console will kill the process and won’t be able to execute any Demo UI’s use cases until executing steps 3 and 4 again.*


7. After opening the UI, go to “Configuration” and configure all the environment.

* On the Enable Edition tab introduce the next `password: incluit`

* After modifing the paths according to your environment and "Save Configuration".


## DEMO UI USE CASE EXECUTION GUIDE


**DRIVER MANAGEMENT**

[TBD]

**ADDING A NEW DRIVER PHOTO** (for Driver Recognition Feature)

1. Go to the sidebar and select "New Driver".
2. Complete "Driver's Name"
3. Upload the "Driver's Picture" according to the image format suggested.
4. Click on "Create New Driver" button.

**ACCESING DRIVER'S BEHAVIOR CLOUD DASHBOARD**

1. Sign in at AWS services for getting the corresponding permission

```
Usuario: intel
Password: Intel2019_
```

2. Go to the sidebar and select "Dashboard".


## TROUBLESHOOTING

When receiving the following error message: **ModuleNotFoundError: No module named ´flask´**

Remember execute the following command inside the ui's folder: `pip3 install -r requirements.txt`