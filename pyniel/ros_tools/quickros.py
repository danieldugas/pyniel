### Tools for quickly getting ros messages

import functools

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
            print("Usage: q = QuickSubscriber(topic_name, msg_class ,n_messages(default=1), verbose(default=False))")
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

class QuickMultiSubscriber(object):
    """ Quickly extracts messages from multiple topics"""
    def __init__(self, subscribers_list=None, n_messages=1, verbose=False):
        if subscribers_list is None:
            print("Usage: q = QuickSubscriber(subscribers_list, n_messages(default=1), verbose(default=False))")
            print("     subscribers_list is a list of dicts, for example:")
            print("     [{'topic': 'my_topic', 'msg_class': sensor_msgs.msg.Image}]")
            raise Exception()
        self.topic_names = [item['topic'] for item in subscribers_list]
        self.messages = [[] for item in subscribers_list]
        self.msg_classes = [item['msg_class'] for item in subscribers_list]
        self.n_messages = n_messages
        self.VERBOSE = verbose
        rospy.init_node("quick_multi_subscriber_{}".format(self.topic_names[0].replace('/','_')), anonymous=True)
        for topic_name, msg_class, messages in zip(self.topic_names, self.msg_classes, self.messages):
            rospy.Subscriber(topic_name, msg_class, functools.partial(self.callback,
                                                                      messages,
                                                                      topic_name))
            print("Waiting for {} messages on topic {}".format(self.n_messages, topic_name))
        rospy.spin()

    def callback(self, messages, topic_name, msg):
        verbose_msg_info = ": {}".format(msg) if self.VERBOSE else ""
        rospy.loginfo("Captured message {}/{}{} for topic {}".format(len(messages),
                                                          self.n_messages,
                                                          verbose_msg_info,
                                                          topic_name))
        messages.append(msg)
        if len(messages) > self.n_messages:
            print("Collected all messages. Messages stored in QuickMultiSubscriber.messages")
            rospy.signal_shutdown("Mission accomplished.")

def subscribers_list_example():
    return [{'topic': "topic_a", 'msg_class': sensor_msgs.msg.Image},
            {'topic': "topic_b", 'msg_class': sensor_msgs.msg.Image}]


