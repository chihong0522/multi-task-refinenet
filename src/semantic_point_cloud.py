import base64
import logging
import time
import roslibpy
import numpy as np
from PIL import Image
import io
import json
import sys
from models import net
import torch, torchvision
from torch.autograd import Variable
import cv2
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import open3d_ros_helper
import rospy
import open3d as o3d
# from rospy_message_converter import message_converter
from sensor_msgs.msg import PointCloud2, PointField, CompressedImage

CMAP = np.load('/home/chihung/multi-task-refinenet/src/cmap_nyud.npy')
DEPTH_COEFF = 5000. # to convert into metres
HAS_CUDA = torch.cuda.is_available()
# HAS_CUDA = None
IMG_SCALE  = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
MAX_DEPTH = 8.
MIN_DEPTH = 0.
NUM_CLASSES = 40
NUM_TASKS = 2 # segm + depth

def prepare_img(img):
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD

model = net(num_classes=NUM_CLASSES, num_tasks=NUM_TASKS)
if HAS_CUDA:
    _ = model.cuda()
_ = model.eval()

ckpt = torch.load('/home/chihung/multi-task-refinenet/weights/ExpNYUD_joint.ckpt')
model.load_state_dict(ckpt['state_dict'])

# Configure logging
# fmt = '%(asctime)s %(levelname)8s: %(message)s'
# logging.basicConfig(format=fmt, level=logging.INFO)
# log = logging.getLogger(__name__)


client = roslibpy.Ros(host='localhost', port=9090)
client.run()

class display_img:
    def __init__(self):
        self.min_max_scaler = MinMaxScaler()
        # self.point_cloud_publisher = roslibpy.Topic(client, 
        #                                             name='/semantic_point_cloud', 
        #                                             message_type='sensor_msgs/PointCloud2')
        rospy.init_node('semantic_point_cloud', anonymous=True)
        self.pointcloud_pub = rospy.Publisher('semantic_point_cloud', PointCloud2, queue_size=20)
        self.header_list = {
            "rgb_image":[],
            "depth_image":[]
        }

    def receive_image(self,msg):
        self.header_list['rgb_image'].append(msg['header']['stamp'])
        start_time = time.time()

        base64_bytes = msg['data'].encode('ascii')
        image_bytes = base64.b64decode(base64_bytes)
        # self.opt_raw_image.value = image_bytes
    
        image = np.array(Image.open(io.BytesIO(image_bytes)))
        
        semantic_frame, depth_frame = self.inference(image)
        
        open3d_pcd = self.create_point_cloud(semantic_frame, depth_frame)
        ros_point_cloud = open3d_ros_helper.o3dpc_to_rospc(open3d_pcd,frame_id="semantic_pcd_frame") #ros point cloud msg
        self.pointcloud_pub.publish(ros_point_cloud)

        end_time = time.time()
        sec_per_infer = end_time - start_time
        fps = 1 / sec_per_infer
        print(f"Published ROS semantic topic FPS = {fps}",end="\r")

    # def show_receive_fps(self,msg):
    #     now_frame =  time.time()
    #     frame_per_sec = now_frame - self.previous_frame 
    #     print(f"receive FPS = {1/ frame_per_sec}",end="\r")
    #     self.previous_frame = now_frame
        
    @staticmethod
    def image_to_byte_array(image:Image):
        imgByteArr = io.BytesIO()
        image.save(imgByteArr, format="jpeg")
        return imgByteArr.getvalue()
        
    def inference(self,img):
        start_time = time.time()
        with torch.no_grad():
            img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2, 0, 1)[None]), requires_grad=False).float()
            if HAS_CUDA:
                img_var = img_var.cuda()
            segm, depth = model(img_var)
            segm = cv2.resize(segm[0, :NUM_CLASSES].cpu().data.numpy().transpose(1, 2, 0),
                              img.shape[:2][::-1],
                              interpolation=cv2.INTER_CUBIC)
            depth = cv2.resize(depth[0, 0].cpu().data.numpy(),
                               img.shape[:2][::-1],
                               interpolation=cv2.INTER_CUBIC)
            segm = CMAP[segm.argmax(axis=2) + 1].astype(np.uint8)
            depth = np.abs(depth)
            
        
        # end_time = time.time()
        # sec_per_infer = end_time - start_time
        # fps = 1 / sec_per_infer
        # print(f"Estimated sementic segmentation FPS = {fps}",end="\r")
        return segm, depth
    
    @staticmethod
    def create_point_cloud(semantic_frame,depth_frame)-> o3d.geometry.PointCloud:
        sem_img = o3d.geometry.Image(np.array(semantic_frame))
        depth_img = o3d.geometry.Image(np.array(depth_frame))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                sem_img, depth_img,convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd.scale(scale=1000,center=np.array([0,0,0]))
        pcd.remove_non_finite_points()
        # pcd.uniform_down_sample(every_k_points=2)
        # pcd.remove_radius_outlier(nb_points=300,radius=100)
        return pcd
    

    def receive_depth(self,msg):
        self.header_list['depth_image'].append(msg['header']['stamp'])
    
try:
    img_stream = display_img()
    
    subscriber = roslibpy.Topic(client, '/camera/color/image_raw/compressed',"sensor_msgs/CompressedImage",throttle_rate=100)
    subscriber.subscribe(img_stream.receive_image)

    sub_2 = roslibpy.Topic(client, '/camera/depth/image_rect_raw/compressed',"sensor_msgs/CompressedImage",throttle_rate=100)
    sub_2.subscribe(img_stream.receive_depth)

    time.sleep(10000)
    
except Exception as e:
    print(e)
    client.terminate()

finally:
    client.terminate()