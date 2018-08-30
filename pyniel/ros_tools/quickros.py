### Tools for quickly getting ros messages

import rospy
from sensor_msgs.msg import *
from std_msgs.msg import *
import numpy as np
from matplotlib import pyplot as plt
import sys

class QuickSubscriber(object):
    """ Quickly extracts messages """
    def __init__(self, topic_name=None, msg_class=Image, n_messages=1, verbose=False):
        if topic_name is None:
            print("Usage: q = QuickSubscriber(topic_name, msg_class ,n_messages(default=1))")
            raise Exception()
        rospy.init_node("quick_subscriber_{}".format(topic_name.replace('/','_')), anonymous=True)
        rospy.Subscriber(topic_name, msg_class, self.callback)
        self.messages = []
        self.n_messages = n_messages
        self.topic_name = topic_name
        self.VERBOSE = verbose
        self.TF_LISTENER = False
        if self.TF_LISTENER:
            self.tfs = []
            self.tf_listener = tf.TransformListener()
        print("Waiting for {} messages on topic {}".format(self.n_messages, self.topic_name))
        rospy.spin()

    def callback(self, msg):
        verbose_msg_info = ": {}".format(msg) if self.VERBOSE else ""
        rospy.loginfo("Captured message {}/{}{}".format(len(self.messages),
                                                          self.n_messages,
                                                          verbose_msg_info));
        self.messages.append(msg)
        if self.TF_LISTENER:
            try:
                time = rospy.Time(0)
                tf = self.tf_listener.lookupTransform(self.frame1, self.frame2, time)
                self.tfs.append(tf)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("tf for time {} not found".format(time))
        if len(self.messages) >= self.n_messages:
            print("Collected all messages. Messages stored in QuickSubscriber.messages")
            rospy.signal_shutdown("Mission accomplished.")




