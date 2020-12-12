

import rospy
import sys
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.msg  
import numpy as np
from PIL import Image
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import CompressedImage

class depth_converter():
    def __init__(self):
        self.pub = rospy.Publisher("/camera/depth/image_rect_raw/numpy",numpy_msg(Floats),queue_size=20)

        self.sub = rospy.Subscriber("/camera/depth/image_rect_raw", 
        sensor_msgs.msg.Image,callback=self.convert_depth_image, queue_size=10)

    def convert_depth_image(self,ros_image):

        bridge = CvBridge()
        depth_image = bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")
        depth_array = np.array(depth_image, dtype=np.float32)
        
        img_header = np.array([float(ros_image.header.stamp.secs), float(ros_image.header.stamp.nsecs)],dtype=np.float32)
        self.pub.publish(img_header)
        self.pub.publish(depth_array)


def main():
    '''Initializes and cleanup ros node'''
    ic = depth_converter()
    rospy.init_node('depth_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"

if __name__ == '__main__':
    main()