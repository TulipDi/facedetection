'''
copy from inferencce_retinaface.py
'''
import os
import cv2
import time
import torch
import common
import numpy as np
from math import ceil
import pycuda.autoinit
import tensorrt as trt
import pycuda.driver as cuda
from itertools import product as product
import argparse

from write_video import write_video
from utils.prior_box import PriorBox
from utils.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm
import models.config as config

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
device = torch.device('cpu')
#device = torch.device('cuda')
resize = 1
save_image = True
parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('--video_path', default='/home/hhd/code/face_detection/a_data/3_S_cut3s.mp4', type=str, help='video_path')
parser.add_argument('--out_path', default='./res_resnet/conf0.95_', type=str, help='save video' )
parser.add_argument('--engine_path', default='./onnx_trt_model/FaceDetector.trt', type=str, help='engine file path' )
parser.add_argument('--backbone', default='resnet', type=str, help='resnet or mobilenet' )
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--vis_thres', default=0.95, type=float, help='visualization_threshold')
args = parser.parse_args()

def get_engine(engine_file):
    # load engine
    if os.path.exists(engine_file):
            print("Reading from engine file")
            with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())


def format_output(output):
    # format output to proper format
    # Loc - output[0] - shape = (1, var/4, 4)
    # Landms - output[1] - shape = (1, var/10, 10)
    # Conf - output[2] - shape = (1, var/2, 2)

    loc = output[0].reshape(1, int(output[0].shape[0]/4), 4)
    landms = output[1].reshape(1, int(output[1].shape[0]/10), 10)
    conf = output[2].reshape(1, int(output[2].shape[0]/2), 2)

    return loc, conf, landms


def preprocess_input(img_raw, device):
    #img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img)
    #img = torch.from_numpy(img).unsqueeze(0)
    #img = img.to(device)

    scale = scale.to(device)

    return img_raw, img, scale, im_height, im_width


def postprocess(priorbox, decode, scale, loc, cfg, conf, landms, img):
    loc, conf, landms = torch.Tensor(loc), torch.Tensor(conf), torch.Tensor(landms)

    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    #prior_data = prior_data.to(device)
    #loc.data = loc.data.to(device)         #============insert by hhd, to insurce prior and loc in the same device
    variance = torch.Tensor(cfg['variance'])
    #variance = variance.to(device)
    boxes = decode(loc.data.squeeze(0).to(device), prior_data.to(device), variance.to(device))
    #boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    
    #boxes = boxes * scale / resize
    #boxes.to(device)
    boxes = boxes * scale 
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    #landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    landms = decode_landm(landms.data.squeeze(0).to(device), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    #landms = landms * scale1 / resize
    landms = landms * scale1 
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.keep_top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    return dets


def detect(engine, context, batch_size, input_img):

    context.set_binding_shape(0, input_img.shape)
    print(context.get_binding_shape(0))

    inputs, outputs, bindings, stream = common.allocate_buffers_dynamic(engine, context)

    inputs[0].host = input_img
    
    output = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    loc, conf, landms = format_output(np.array(output))
    return loc, conf, landms

def main():
    batch_size = 1

    if args.backbone is 'resnet':
        cfg = config.cfg_re50
    else:
        cfg = config.cfg_mnet

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get engine
    engine = get_engine(args.engine_path)
    context = engine.create_execution_context()
    context.active_optimization_profile = 0
    
    # tools for save video
    out = write_video(result_path=args.out_path,result_name='3_S_cut.mp4',out_fps=30,width=1920,height=1080)

    # testing begin
    cap = cv2.VideoCapture(args.video_path)
    while True:
        tic = time.time()
        _, frame = cap.read()
        if frame is None:
            break
        img_raw, input_img, scale, im_height, im_width = preprocess_input(frame, device)

        # do forwart and test inference time
        #torch.cuda.synchronize()
        #tic = time.time()
        loc, conf, landms = detect(engine, context, batch_size, input_img)
        #torch.cuda.synchronize()
        #print('net forward time: {:.4f}'.format(time.time() - tic))
        
        # postprocessing dependencies
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))

        # post process
        input_img = torch.from_numpy(input_img)
        input_img =input_img.to(device)
        dets = postprocess(priorbox, decode, scale, loc, cfg, conf, landms, input_img)
        
        print('net forward time: {:.4f}'.format(time.time() - tic))

        # save image
        if save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image

            out.write(frame)
    out.release()

if __name__ == '__main__':
    
    main()



