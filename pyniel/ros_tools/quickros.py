### Tools for quickly getting ros messages

import functools

import rospy
import tf
from tf2_ros import TransformException
from sensor_msgs.msg import *
from std_msgs.msg import *
import numpy as np
from matplotlib import pyplot as plt
import sys

class QuickSubscriber(object):
    """ Quickly extracts messages """
    def __init__(self, topic_name=None, msg_class=Image, n_messages=1, verbose=False,
            tf_parent_frame=None, tf_timeout=1., tf_skip_if_no_tf=True):
        if topic_name is None:
            print("Usage: q = QuickSubscriber(topic_name, msg_class ,n_messages(default=1), verbose(default=False))")
            raise Exception()
        rospy.init_node("quick_subscriber_{}".format(topic_name.replace('/','_')), anonymous=True)
        rospy.Subscriber(topic_name, msg_class, self.callback)
        self.messages = []
        self.n_messages = n_messages
        self.topic_name = topic_name
        self.VERBOSE = verbose
        self.TF_PARENT_FRAME = tf_parent_frame
        self.tf_timeout = rospy.Duration(tf_timeout)
        self.tf_skip_if_tf_not_found = tf_skip_if_no_tf
        if self.TF_PARENT_FRAME is not None:
            self.tfs = []
            self.tf_listener = tf.TransformListener()
        print("Waiting for {} messages on topic {}".format(self.n_messages, self.topic_name))
        rospy.spin()

    def callback(self, msg):
        verbose_msg_info = ": {}".format(msg) if self.VERBOSE else ""
        rospy.loginfo("Captured message {}/{}{}".format(len(self.messages)+1,
                                                          self.n_messages,
                                                          verbose_msg_info));
        if self.TF_PARENT_FRAME is not None:
            try:
                time = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
                tf_info = [self.TF_PARENT_FRAME, msg.header.frame_id, time]
                self.tf_listener.waitForTransform(*(tf_info + [self.tf_timeout]))
                tf_ = self.tf_listener.lookupTransform(*tf_info)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException,
                    TransformException) as e:
                print("tf for time {} not found: {}".format(time, e))
                if self.tf_skip_if_tf_not_found:
                    print("Skipping message.")
                    return
                else:
                    tf_ = None
            self.tfs.append(tf_)
        self.messages.append(msg)
        if len(self.messages) >= self.n_messages:
            print("Collected all messages. Messages stored in QuickSubscriber.messages")
            if self.TF_PARENT_FRAME is not None:
                print("tfs stored in QuickSubscriber.tfs")
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


