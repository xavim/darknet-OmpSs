#!python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Both the GPU and no-GPU version should be compiled; the no-GPU version should be renamed "yolo_cpp_dll_nogpu.dll".

On a GPU system, you can force CPU evaluation by any of:

- Set global variable DARKNET_FORCE_CPU to True
- Set environment variable CUDA_VISIBLE_DEVICES to -1
- Set environment variable "FORCE_CPU" to "true"


To use, either run performDetect() after import, or modify the end of this file.

See the docstring of performDetect() for parameters.

Directly viewing or returning bounding-boxed images requires scikit-image to be installed (`pip install scikit-image`)


Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn
@date: 20180503
"""
#pylint: disable=R, W0401, W0614, W0703
from ctypes import *
import math
import random
import os

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("darknet.so", RTLD_GLOBAL)

lib = CDLL("./darknet.so", RTLD_GLOBAL)

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

array_to_image = lib.array_to_image
array_to_image.argtypes = [POINTER(c_char),c_int,c_int,c_int]
array_to_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

""" Moved to C (image.c/image.h)!!!
def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr
"""

if __name__ == "__main__":
	import cv2
	import time
	print("starting camera yolo example")

	cv2.namedWindow("image", cv2.WINDOW_NORMAL)
	cap = cv2.VideoCapture(3)
	cap.set(3,1920);
	cap.set(4,1080);

	configPath = "./cfg/yolov3.cfg"
	weightPath = "yolov3.weights"
	metaPath = "./cfg/coco.data"

	thresh = 0.3
	hier_thresh=.5
	nms=.45 
	debug= False

	net = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
	meta = load_meta(metaPath.encode("ascii"))


	c_char_p = POINTER(c_char)

	while True:

		start_time = time.time()

		ret, frame = cap.read()
		if ret is False:
			continue

		#data = image.astype(numpy.int8)
		data_p = frame.ctypes.data_as(c_char_p)

		im = array_to_image(data_p, frame.shape[0],frame.shape[1],frame.shape[2])

		num = c_int(0)
		pnum = pointer(num)
	
		predict_image(net, im)
		dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)

		num = pnum[0]
		if nms:
			do_nms_sort(dets, num, meta.classes, nms)

		res = []
	
		for j in range(num):
			for i in range(meta.classes):
				if dets[j].prob[i] > 0:
					b = dets[j].bbox
					nameTag = meta.names[i]

					cv2.circle(frame , (int(b.x), int(b.y))  , 5,(255,255,255), 5)
					cv2.rectangle(frame,(int(b.x-b.w/2),int(b.y-b.h/2)),(int(b.x+b.w/2),int(b.y+b.h/2)) ,(55,255,55), 3)


		free_image(im)
		free_detections(dets, num)


		delta = time.time() - start_time
		fps = 1. / delta

		cv2.putText(frame, str(fps) + " FPS", (50, 100), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(50,255,50), thickness=3)
		
		cv2.imshow("image", frame)
	
		k = cv2.waitKey(33)
		if k == ord('q'):
			exit()


