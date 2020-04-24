import subprocess
import configparser
from subprocess import Popen, PIPE
from flask import Flask, render_template, redirect, url_for, request, jsonify, Response
import os
import shutil
import hashlib
import time
from glob import glob

# ----- CONFIGURATION -----

# Constant Variables
OPENVINO_SOURCE = '/opt/intel/openvino/bin/setupvars.sh'
ROS_SOURCE = '/opt/ros/crystal/setup.bash'

# Variables from file
config = configparser.SafeConfigParser()
config.read('config.ini')  # Take The parameters from config.ini file
# Driver Management (ROS2 Workspace)
workspace = os.path.join(config['workspace']['path'], '')

# Variables Initialization
dirname = os.path.dirname(__file__)
file_input = os.path.join(dirname, 'tmp', '')

aws_folder = workspace + "AWS/"
driverbehavior_folder = workspace + "DriverBehavior/"
actionrecognition_folder = workspace + "ActionRecognition/"

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
print(' * Myriad Detected: ' + str(myriad))


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
        command_rosbag = ("source " + ROS_SOURCE + " && source " + driverbehavior_folder + "ets_ros2/install/setup.bash && cd " +
                          driverbehavior_folder + " && while true; do ros2 bag play truck.bag; done;") if (json['rosbag'] == "1") else ("")

        # Driver Actions Command
        command_driver_actions = "source " + ROS_SOURCE + " && source " + OPENVINO_SOURCE + \
            " && cd " + actionrecognition_folder + " && python3 action_recognition.py -m_en models/FP32/driver-action-recognition-adas-0002-encoder.xml -m_de models/FP32/driver-action-recognition-adas-0002-decoder.xml -lb driver_actions.txt -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU"

        if (json['camera_actions'] == "0"):
            command_driver_actions += " -i '" + \
                file_input + json['file_actions'] + "'"
        else:
            if (json['camera'] == "0"):
                command_driver_actions += " -i /dev/video1"
            else:
                command_driver_actions += " -i /dev/video0"

        if (json['aws_actions']):
            command_driver_actions += " -e a1572pdc8tbdas-ats.iot.us-east-1.amazonaws.com -r " + aws_folder + "AmazonRootCA1.pem -c " + \
                aws_folder + "a81867df13-certificate.pem.crt -k " + \
                aws_folder + "a81867df13-private.pem.key -t actions/"

        # Driver Behaviour Command
        command_driver_behaviour = "source " + ROS_SOURCE + " && source " + driverbehavior_folder + "ets_ros2/install/setup.bash && source " + OPENVINO_SOURCE + " && source " + driverbehavior_folder + "scripts/setupenv.sh && cd " + driverbehavior_folder + "build/intel64/Release && ./driver_behavior -d " + \
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
        # lib CPU extension
        command_driver_behaviour += " -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so "
        # Recognition of the Driver
        if (json['recognition'] == "1"):
            command_driver_behaviour += " -d_recognition -fg " + \
                driverbehavior_folder + "scripts/faces_gallery.json"
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
        if (json['camera_actions'] == "0"):
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
            file.save(os.path.join(driverbehavior_folder + "drivers/",
                                   request.values['driver'] + "." + file.filename.split('.')[-1]))
            # Generating the list with all the drivers
            print("Creating New Driver")
            shell_communication("cd " + driverbehavior_folder +
                                "scripts/ && python3 create_list.py ../drivers/")
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
    drivers_path = driverbehavior_folder + "/drivers/"
    driver_list = [f for f in os.listdir(os.path.join(
        drivers_path)) if os.path.isfile(os.path.join(drivers_path, f))]
    driver_list.sort()  # Order the list by name

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
        'workspace': workspace
    }
    return render_template("configuration.html", **templateData)


@app.route("/certificates")
def certificates():
    templateData = {  # Sending the data to the frontend
        'title': "Certificates Configuration",
        'workspace': workspace
    }
    return render_template("certificates.html", **templateData)


@app.route('/upload_certificates', methods=['POST', 'GET'])
# Upload the video file selected into a temporal folder (Video Upload Folder in configuration)
def upload_certificates():
    out = False
    if request.method == 'POST':
        if request.files:
            # Certificate
            certificate = request.files.get('certificate', False)
            if certificate:
                file = request.files["certificate"]
                file.save(os.path.join(aws_folder, file.filename))
            # Private Key
            private_key = request.files.get('private_key', False)
            if private_key:
                file = request.files["private_key"]
                file.save(os.path.join(aws_folder, file.filename))
            # Public Key
            public_key = request.files.get('public_key', False)
            if public_key:
                file = request.files["public_key"]
                file.save(os.path.join(aws_folder, file.filename))
            # RootCA
            root_ca = request.files.get('root_ca', False)
            if root_ca:
                file = request.files["root_ca"]
                file.save(os.path.join(aws_folder, file.filename))
            out = True
    return jsonify(out=out)


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

        config['workspace']['path'] = json['workspace']

        # Saving variables in config.ini
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
            out = True

        # Variables are updated with the new values.
        global workspace
        # Workspace
        workspace = os.path.join(config['workspace']['path'], '')

    return jsonify(out=out)


app.run(debug=True)  # Run Flask App
