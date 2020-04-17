#!/usr/bin/env python
"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import print_function

import sys
from argparse import ArgumentParser, SUPPRESS
import signal
import os
import sys
import time

from openvino.inference_engine import IECore

from include.models import IEModel
from include.result_renderer import ResultRenderer
from include.steps import run_pipeline
from include.config import *
#import boto3
#from botocore.exceptions import NoCredentialsError
from include.basicpubsub import BasicPubSub
from os import path

"""
# Send data to S3
ACCESS_KEY = 'XXXXXXXXXXXXXXXX'
SECRET_KEY = 'XXXXXXXX/XXXXXXXXXXXXXXXXXXXXXXXX'
AllowedActions = ['both', 'publish', 'subscribe']

def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)
    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
"""
state = {"signal": False, "ready": True }
AllowedActions = ['both', 'publish', 'subscribe']

def video_demo(encoder, decoder, videos, fps=30, labels=None, publicAWS=None):
    """Continuously run demo on provided video list"""
    result_presenter = ResultRenderer(labels=labels, publicAWS = publicAWS)
    run_pipeline(videos, encoder, decoder, result_presenter.render_frame, fps=fps)

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m_en", "--m_encoder", help="Required. Path to encoder model", required=True, type=str)
    args.add_argument("-m_de", "--m_decoder", help="Required. Path to decoder model", required=True, type=str)
    args.add_argument("-i", "--input",
                      help="Required. Path to a video or a .txt file with a list of video files (one video per line)", type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. For CPU custom layers, if any. Absolute path to a shared library with the "
                           "kernels implementation.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify a target device to infer on. CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for the device specified. "
                           "Default value is CPU",
                      default="CPU", type=str)
    args.add_argument("--fps", help="Optional. FPS for renderer", default=30, type=int)
    args.add_argument("-lb", "--labels", help="Optional. Path to file with label names", type=str)
    # Data to AWS
    args.add_argument("-e", "--endpoint", action="store", dest="host", help="Your AWS IoT custom endpoint")
    args.add_argument("-r", "--rootCA", action="store", dest="rootCAPath", help="Root CA file path")
    args.add_argument("-c", "--cert", action="store", dest="certificatePath", help="Certificate file path")
    args.add_argument("-k", "--key", action="store", dest="privateKeyPath", help="Private key file path")
    args.add_argument("-p", "--port", action="store", dest="port", type=int, help="Port number override")
    args.add_argument("-w", "--websocket", action="store_true", dest="useWebsocket", default=False,
                        help="Use MQTT over WebSocket")
    args.add_argument("-id", "--clientId", action="store", dest="clientId", default="basicPubSub",
                        help="Targeted client id")
    args.add_argument("-t", "--topic", action="store", dest="topic", default="sdk/test/Python", help="Targeted topic")
    args.add_argument("-m", "--mode", action="store", dest="mode", default="both",
                        help="Operation modes: %s"%str(AllowedActions))
    args.add_argument("-M", "--message", action="store", dest="message", default="Hello World!",
                        help="Message to publish")
    args.add_argument("--only", help="Optional. Run Driver Action without Driver Assistance waiting signal", dest="only", action='store_true')                        
    return parser

def receiveSignal(signalNumber, frame):
    global state
    if (state["ready"]):
        state["signal"] = True
    return

def terminateProcess(signalNumber, frame):
    print ('(SIGTERM) terminating the process')
    sys.exit()

def main():
    args = build_argparser().parse_args()

    signal.signal(signal.SIGUSR1, receiveSignal)
    signal.signal(signal.SIGINT, terminateProcess)
    signal.signal(signal.SIGTERM, terminateProcess)

    if args.rootCAPath is not None:
        # Data to AWS  
        if args.mode not in AllowedActions:
            args.error("Unknown --mode option %s. Must be one of %s" % (args.mode, str(AllowedActions)))
            exit(2)
        
        if args.useWebsocket and args.certificatePath and args.privateKeyPath:
            args.error("X.509 cert authentication and WebSocket are mutual exclusive. Please pick one.")
            exit(2)

        if not args.useWebsocket and (not args.certificatePath or not args.privateKeyPath):
            args.error("Missing credentials for authentication.")
            exit(2)

        publicAWS = BasicPubSub(host = args.host, rootCAPath = args.rootCAPath, certificatePath = args.certificatePath, privateKeyPath = args.privateKeyPath, port = args.port, useWebsocket = args.useWebsocket, clientId = args.clientId, topic = args.topic, mode = args.mode, message = args.mode)
        publicAWS.suscribeMQTT()
        # End Data to AWS

    full_name = path.basename(args.input)
    extension = path.splitext(full_name)[1]

    if '.txt' in  extension:
        with open(args.input) as f:
            videos = [line.strip() for line in f.read().split('\n')]
    else:
        videos = [args.input]

    if not args.input:
        raise ValueError("--input option is expected")

    if args.labels:
        with open(args.labels) as f:
            labels = [l.strip() for l in f.read().strip().split('\n')]
    else:
        labels = None

    ie = IECore()

    if 'MYRIAD' in args.device:
        myriad_config = {"VPU_HW_STAGES_OPTIMIZATION": "YES"}
        ie.set_config(myriad_config, "MYRIAD")

    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")

    decoder_target_device = "CPU"
    if args.device != 'CPU':
        encoder_target_device = args.device
    else:
        encoder_target_device = decoder_target_device

    encoder_xml = args.m_encoder
    encoder_bin = args.m_encoder.replace(".xml", ".bin")
    decoder_xml = args.m_decoder
    decoder_bin = args.m_decoder.replace(".xml", ".bin")

    encoder = IEModel(encoder_xml, encoder_bin, ie, encoder_target_device,
                      num_requests=(3 if args.device == 'MYRIAD' else 1))
    decoder = IEModel(decoder_xml, decoder_bin, ie, decoder_target_device, num_requests=2)
    print("Waiting on signal")
    while (True):
        time.sleep(1)
        if (state["signal"]):
            state["signal"] = False
            state["ready"] = False
            #upload_to_aws('README.md', 'driver-actions', 'README100.md')
            if args.rootCAPath:
                video_demo(encoder, decoder, videos, args.fps, labels, publicAWS)
            else:
                video_demo(encoder, decoder, videos, args.fps, labels)
            state["ready"] = True

if __name__ == '__main__':
    sys.exit(main() or 0)
