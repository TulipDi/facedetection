import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from write_video import write_video
from data.config import cfg_re50
from data.config import cfg_mnet
#from models.cfg import cfg_mnet
from models.retinaface import RetinaFace
from utils.nms.py_cpu_nms import py_cpu_nms
#from utils.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm
from layers.functions.prior_box import PriorBox
#from utils.prior_box import PriorBox
import cv2
import time

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('--video_path', default='/home/hhd/code/face_detection/a_data/3_S_cut3s.mp4', type=str, help='video_path')
#parser.add_argument('--out_path', default='./res_mobilenet/conf0.95_', type=str, help='save video' )
parser.add_argument('--out_path', default='./res_resnet/conf0.95_', type=str, help='save video' )
#parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--vis_thres', default=0.95, type=float, help='visualization_threshold')
args = parser.parse_args()
#os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def detect_face(net, img):

    img = img.astype(np.float32)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    print(img.shape)
    #torch.cuda.synchronize()
    #tic = time.time()
    loc, conf, landms = net(img)  # forward pass
    #torch.cuda.synchronize()
    #print('net forward time: {:.4f}.'.format(time.time() - tic))

    resize = 1
    #priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
    priorbox = PriorBox(cfg_re50, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    #boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
    boxes = decode(loc.data.squeeze(0), prior_data, cfg_re50['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    #landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_re50['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
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
    

if __name__ == '__main__':

    # net and model
    #net = RetinaFace(cfg=cfg_mnet, phase = 'test')
    net = RetinaFace(cfg=cfg_re50, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    
    #device = torch.device("cpu" if args.cpu else "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # tools for save video 
    out = write_video(result_path=args.out_path,result_name='3_S_cut.mp4',out_fps=30,width=1920,height=1080)

    # testing begin
    cap = cv2.VideoCapture(args.video_path)
    while True:
        tic = time.time()
        _, frame = cap.read()
        if frame is None:
            break
         
        #frame = cv2.resize(frame, (300, 300))
        dets = detect_face(net, frame)
        print(len(dets))

        print('net forward time: {:.4f}.'.format(time.time() - tic))
        # draw info on image
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # draw landmarks with different color
            cv2.circle(frame, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(frame, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(frame, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(frame, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(frame, (b[13], b[14]), 1, (255, 0, 0), 4)


        # save video
        out.write(frame)
        
        # show image
        # cv2.imshow("test", frame)
        # cv2.waitKey(1)
    out.release()
