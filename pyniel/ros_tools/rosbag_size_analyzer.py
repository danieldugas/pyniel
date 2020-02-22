from __future__ import print_function
import rosbag
import sys

rosbag_path = sys.argv[1]
messages = rosbag.Bag(rosbag_path, 'r').read_messages(raw=True)

topic_size_dict = {}
for topic, msg, time in messages:
    msg_size = len(msg[1])
    topic_size_dict[topic] = topic_size_dict.get(topic, 0) + msg_size
topic_size = list(topic_size_dict.items())
topic_size.sort(key=lambda x: x[1])
for topic, size in topic_size[::-1]:
    print(topic, size / 1000000., "MB")
print("----------------------------")
print("total: ", sum([size for _, size in topic_size]))
