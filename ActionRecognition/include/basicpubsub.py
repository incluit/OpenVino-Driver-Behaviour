'''
/*
 * Copyright 2010-2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *  http://aws.amazon.com/apache2.0
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */
 '''

from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import logging
import time
import argparse
import json

class BasicPubSub(object):
    def __init__(self, host = None, rootCAPath = None, certificatePath = None, privateKeyPath = None, port = None, useWebsocket = None, clientId = None, topic = None, mode = None, message = None):

        self.host = host
        self.rootCAPath = rootCAPath
        self.certificatePath = certificatePath
        self.privateKeyPath = privateKeyPath
        self.port = port
        self.useWebsocket = useWebsocket
        self.clientId = clientId
        self.topic = topic
        self.mode = mode
        self.message = message

        # Port defaults
        if self.useWebsocket and not self.port:  # When no port override for WebSocket, default to 443
            port = 443
        if not self.useWebsocket and not self.port:  # When no port override for non-WebSocket, default to 8883
            port = 8883

        # Configure logging
        logger = logging.getLogger("AWSIoTPythonSDK.core")
        logger.setLevel(logging.DEBUG)
        streamHandler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        streamHandler.setFormatter(formatter)
        logger.addHandler(streamHandler)

        # Init AWSIoTMQTTClient
        self.myAWSIoTMQTTClient = None
        if useWebsocket:
            self.myAWSIoTMQTTClient = AWSIoTMQTTClient(clientId, useWebsocket=True)
            self.myAWSIoTMQTTClient.configureEndpoint(host, port)
            self.myAWSIoTMQTTClient.configureCredentials(rootCAPath)
        else:
            self.myAWSIoTMQTTClient = AWSIoTMQTTClient(clientId)
            self.myAWSIoTMQTTClient.configureEndpoint(host, port)
            self.myAWSIoTMQTTClient.configureCredentials(rootCAPath, privateKeyPath, certificatePath)

        # AWSIoTMQTTClient connection configuration
        self.myAWSIoTMQTTClient.configureAutoReconnectBackoffTime(1, 32, 20)
        self.myAWSIoTMQTTClient.configureOfflinePublishQueueing(-1)  # Infinite offline Publish queueing
        self.myAWSIoTMQTTClient.configureDrainingFrequency(2)  # Draining: 2 Hz
        self.myAWSIoTMQTTClient.configureConnectDisconnectTimeout(10)  # 10 sec
        self.myAWSIoTMQTTClient.configureMQTTOperationTimeout(5)  # 5 sec
        
    # Custom MQTT message callback
    def customCallback(self, client, userdata, message):
        print("Received a new message: ")
        print(message.payload)
        print("from topic: ")
        print(message.topic)
        print("--------------\n\n")

    # Connect and subscribe to AWS IoT
    def suscribeMQTT(self):
        self.myAWSIoTMQTTClient.connect()
        if self.mode == 'both' or self.mode == 'subscribe':
            self.myAWSIoTMQTTClient.subscribe(self.topic, 1, self.customCallback)
        time.sleep(2)

    def publicMQTT(self, action = None, action_prob = None):
        if self.mode == 'both' or self.mode == 'publish':
            message = {}
            message['timestamp'] = time.time()*1000
            message['action'] = action
            message['action_prob'] = action_prob
            messageJson = json.dumps(message)
            self.myAWSIoTMQTTClient.publish(self.topic, messageJson, 1)
            if self.mode == 'publish':
                print('Published topic %s: %s\n' % (self.topic, messageJson))
        #time.sleep(1)