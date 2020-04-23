import subprocess
import configparser
from subprocess import Popen, PIPE
from flask import Flask, render_template, redirect, url_for, request, jsonify, Response
import os
import shutil
import hashlib
import time
from glob import glob

# ----- Configuration -----
config = configparser.SafeConfigParser()
config.read('config.ini')  # Take all the parameters from config.ini file
# Local
dirname = os.path.dirname(__file__)
file_input = os.path.join(dirname, 'tmp', '')
# OpenVino
openvino_source = config['openvino']['source'].rstrip('/')
# Driver Management (ROS2 Workspace)
ros_path = os.path.join(config['ros']['path'], '')
# Driver Management (Repository Folder)
dmanagement_repository = os.path.join(config['management']['path'])


def shell_communication(cmd):
    # This function allows to execute a bash command
    session = subprocess.Popen(
        [cmd], stdout=PIPE, stderr=PIPE, shell=True, executable="/bin/bash")
    stdout, stderr = session.communicate()
    if stderr:
        raise Exception("Error "+str(stderr))
    return stdout.decode('utf-8')


def shell_communication_parallel(cmds):
    # --- Running in Parallel ---
    # Rosbag
    print(" --- Initializing Driver Management --- ")
    print("Loading Rosbag")
    rosbag = Popen(cmds[0], stdout=None, stderr=None,
                   shell=True, executable="/bin/bash")
    # Driver Actions
    print("Loading Driver Actions")
    driver_actions = Popen(cmds[1], stdout=None, stderr=None,
                           shell=True, executable="/bin/bash")
    # Driver Behaviour
    print("Loading Driver Behaviour")
    driver_behaviour = Popen(cmds[2] + str(driver_actions.pid), stdout=None, stderr=None,
                             shell=True, executable="/bin/bash")

    print(" --- Ready! ---")
    rosbag.wait()
    driver_actions.wait()
    driver_behaviour.wait()


app = Flask(__name__)  # Flask constructor

# Check if there are MDX (MyriadX) or NCS (Neural Compute Stick).
try:
    subprocess.check_output('dmesg | grep Myriad', shell=True)
    myriad = True
except:
    myriad = False
print('Myriad Detected: ' + str(myriad))


# Wait until 10 seconds to check if a file exists.
def wait_for_file(file):
    print("Uploading file...")
    time_counter = 0
    while not (os.path.exists(file)):
        time.sleep(1)
        time_counter += 1
        if time_counter > 10:
            break

# ----- Route Definitions -----
@app.route("/", methods=['POST', 'GET'])
def home():
    templateData = {  # Sending the data to the frontend
        'title': "Driver Management",
        'myriad': myriad
    }
    return render_template("driver-management.html", **templateData)


@app.route('/upload_file', methods=['POST', 'GET'])
# Upload the video file selected into a temporal folder (Video Upload Folder in configuration)
def upload_file():
    if request.method == 'POST':
        if request.files:
            file_driver = request.files.get('file', False)
            if file_driver:
                file = request.files["file"]
                file.save(os.path.join(file_input, file.filename))
            # Detect if exists file_actions
            file_actions = request.files.get('file_actions', False)
            if file_actions:
                file = request.files["file_actions"]
                file.save(os.path.join(file_input, file.filename))
            # End check file_actions
        return ''


@app.route('/run_driver_management', methods=['POST', 'GET'])
# This function runs the bash command when the user runs Driver Management Project in the interface.
def run_driver_management():
    if request.method == 'POST':
        json = request.get_json()

        # Rosbag Command
        command_rosbag = ("source /opt/ros/crystal/setup.bash && source " + ros_path + "install/setup.bash && cd " +
                          ros_path + " && while true; do ros2 bag play truck.bag; done;") if (json['rosbag'] == "1") else ("")

        # Driver Actions Command
        command_driver_actions = "source /opt/ros/crystal/setup.bash && source /app/DriverBehavior/ets_ros2/install/setup.bash && source /opt/intel/openvino/bin/setupvars.sh && cd /app/ActionRecognition && python3 action_recognition.py -m_en models/FP32/driver-action-recognition-adas-0002-encoder.xml -m_de models/FP32/driver-action-recognition-adas-0002-decoder.xml -lb driver_actions.txt -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU"

        if (json['camera_actions'] == "0"):
            command_driver_actions += " -i '" + \
                file_input + json['file_actions'] + "'"
        else:
            command_driver_actions += " -i /dev/video1"

        if (json['aws_actions']):
            command_driver_actions += " -e a1572pdc8tbdas-ats.iot.us-east-1.amazonaws.com -r aws-certificates/AmazonRootCA1.pem -c aws-certificates/a81867df13-certificate.pem.crt -k aws-certificates/a81867df13-private.pem.key -t actions/"

        # Driver Behaviour Command
        command_driver_behaviour = "source /opt/ros/crystal/setup.bash && source /app/DriverBehavior/ets_ros2/install/setup.bash && source /opt/intel/openvino/bin/setupvars.sh && source /app/" + dmanagement_repository + "/scripts/setupenv.sh && cd /app/" + dmanagement_repository + "/build/intel64/Release && ./driver_behavior -d " + \
            json['target'] + " -d_hp " + json['target_hp']

        if (json['camera'] == "0"):
            command_driver_behaviour += " -i '" + \
                file_input + json['file'] + "'"
        else:
            command_driver_behaviour += " -i cam"

        if (json['model'] == "face-detection-adas-0001"):
            if (json['precision'] == "FP16"):
                command_driver_behaviour += " -m $face116"
            else:
                command_driver_behaviour += " -m $face132"
        else:
            if (json['model'] == "face-detection-adas-0004"):
                if (json['precision'] == "FP16"):
                    command_driver_behaviour += " -m $face216"
                else:
                    command_driver_behaviour += " -m $face232"
        # Recognition of the Driver
        if (json['recognition'] == "1"):
            command_driver_behaviour += " -d_recognition -fg ../../../../../../src/ets_ros2/aws-crt-cpp/samples/" + \
                dmanagement_repository + "/scripts/faces_gallery.json"
        # Landmarks Detection
        if (json['landmarks'] == "1"):
            command_driver_behaviour += " -dlib_lm"
        # Headpose Detection
        if (json['head_pose'] == "1"):
            command_driver_behaviour += " -m_hp $hp32"
        # Synchronous / Asynchronous mode
        if (json['async'] == "1"):
            command_driver_behaviour += " -async"

        command_driver_behaviour += " -pid_da "

        commands = [command_rosbag, command_driver_actions,
                    command_driver_behaviour]
        if (json['camera'] == "0"):
            wait_for_file(file_input + json['file'])
            wait_for_file(file_input + json['file_actions'])
        print("Running Driver Management")
        shell_communication_parallel(cmds=commands)
        return ("Finish Driver Management")


def killProcess(processes):
    if (type(processes) == list):
        print(" --- Killing Processes --- ")
        for process in processes:
            os.system('pkill -f ' + process)
            print('Procces killed: ' + process)
        print("--- Finish Killing Processes --- ")
        return "The processes were killed correctly!"
    else:
        return "Error trying kill the processes"


@app.route('/stop_driver_management', methods=['POST', 'GET'])
# This function stop the bash command when the user runs Driver Behaviour Project in the interface.
def stop_driver_management():
    processes = [
        # If select "ros" mat be close the proccess of this program too (Because probably is inside the folder "ros2_ws")
        'truck.bag',
        'action_recognition.py',
        'driver_behavior'
    ]
    out = killProcess(processes)
    return jsonify(out=out)


@app.route('/new-driver-management')
def newdriver_management():
    templateData = {  # Sending the data to the frontend
        'title': "New Driver"
    }
    return render_template("new-driver-management.html", **templateData)


@app.route('/create_driver_management', methods=['POST', 'GET'])
# This function allows create a new driver to Driver Behavior.
def create_driver_management():
    if request.method == 'POST':
        out = "The driver couldn't be created"
        if request.files:
            file = request.files["file"]
            # Save the file with the Driver's name and add the same extension
            file.save(os.path.join(ros_path + "/src/ets_ros2/aws-crt-cpp/samples/OpenVino-Driver-Behaviour/" + dmanagement_repository +
                                   "/drivers/", request.values['driver'] + "." + file.filename.split('.')[-1]))
            # Generating the list with all the drivers
            print("Creating New Driver")
            shell_communication("cd " + ros_path + "/src/ets_ros2/aws-crt-cpp/samples/" +
                                dmanagement_repository + "/scripts/ && python3 create_list.py ../drivers/")
            out = "New driver created!"
        return jsonify(out=out)


@app.route('/dashboard')
def dashboard():
    templateData = {  # Sending the data to the frontend
        'title': "Dashboard"
    }
    return render_template("dashboard.html", **templateData)


@app.route('/drivers')
def drivers():
    drivers_path = ros_path + "src/ets_ros2/aws-crt-cpp/samples/OpenVino-Driver-Behaviour/" + dmanagement_repository + "/drivers/"
    driver_list = [f for f in os.listdir(os.path.join(drivers_path)) if os.path.isfile(os.path.join(drivers_path, f))]
    driver_list.sort() # Order the list by name
    
    # Copy folder to static/drivers !!!
    
    templateData = {  # Sending the data to the frontend
        'title': "Drivers",
        'drivers': driver_list,
        'path': os.path.join(drivers_path)
    }
    return render_template("drivers.html", **templateData)


@app.route("/config")
def configuration():
    templateData = {  # Sending the data to the frontend
        'title': "Configuration",
        'openvino_source': openvino_source,
        'ros_path': ros_path,
        'dmanagement_repository': dmanagement_repository
    }
    return render_template("configuration.html", **templateData)


@app.route("/certificates")
def certificates():
    templateData = {  # Sending the data to the frontend
        'title': "Certificates Configuration",
        'openvino_source': openvino_source,
        'ros_path': ros_path,
        'dmanagement_repository': dmanagement_repository
    }
    return render_template("certificates.html", **templateData)


@app.route('/check_pass', methods=['POST', 'GET'])
# This function checks the password to enable the edition of the configuration
def check_pass():
    if request.method == 'POST':
        out = False
        json = request.get_json()
        password = hashlib.md5(json['password'].encode())
        if (password.hexdigest() == "9093363f8ee6138f7ba43606fdab7176"):
            out = True
    return jsonify(out=out)


@app.route('/change_config', methods=['POST', 'GET'])
# This functions saves the new configuration in the config.ini file.
def change_config():
    if request.method == 'POST':
        out = False
        json = request.get_json()

        config['openvino']['source'] = json['openvino_source']
        config['ros']['path'] = json['ros_path']
        config['management']['path'] = json['dmanagement_repository']

        # Saving variables in config.ini
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
            out = True

        # Variables are updated with the new values.
        global openvino_source, ros_path, dmanagement_repository
        # Driver Management (ROS2 Wokspace)
        ros_path = os.path.join(config['ros']['path'], '')
        # Driver Management (Repository Folder)
        dmanagement_repository = os.path.join(config['management']['path'])

    return jsonify(out=out)


app.run(debug=True)  # Run Flask App
