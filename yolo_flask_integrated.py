from flask import Flask, request, jsonify, request, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS

import argparse
import time
import cv2
import threading
import torch
import uuid
import torch.backends.cudnn as cudnn

from datetime import datetime
from pathlib import Path
from numpy import random
from torchvision import transforms
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, non_max_suppression_kpt
from utils.plots import plot_one_box, output_to_keypoint, plot_skeleton_kpts
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# Set
app = Flask(__name__)
CORS(app)  # Allow cross-domain requests
socketio = SocketIO(app, cors_allowed_origins='*')  # Allow all domain cross-domain requests

# store models
global global_opt
global global_obj_model, global_pose_model, global_vio_model  # yolov7; yolov7_w6_pose; yolov7_vio_detect_v3
global_result = {}
result_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    # Setting the device directly to CUDA device 0
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    return parser.parse_args()


# Parse arguments in the global scope
global_opt = parse_args()


# Function to load object detection model
def load_obj_model(weights_path, device):
    model = attempt_load(weights_path, map_location=device)  # load FP32 model
    model.to(device).eval()
    return model.half()  # to FP16


# Load pose model
def load_pose_model(weights_path, device):
    model = torch.load(weights_path, map_location=device)['model'].float().eval()
    model = model.to(device)  # to FP16
    return model


# Load violence detection model
def load_vio_model(weights_path, device):
    model = attempt_load(weights_path, map_location=device)  # load FP32 model
    model.to(device).eval()
    return model.half()  # to FP16


def load_models():
    global global_obj_model, global_pose_model, global_vio_model
    # Explicitly setting device to CUDA device 0
    device = torch.device('cuda:0')
    global_obj_model = load_obj_model('yolov7.pt', device)
    global_pose_model = load_pose_model('yolov7-w6-pose.pt', device)
    global_vio_model = load_vio_model('yolov7_vio_detect_v3.pt', device)


# Load models on application startup
load_models()


# Pose estimation function
def run_pose_estimation(tensor_image, model):
    # Adjust tensor type
    tensor_image = tensor_image.half() if next(model.parameters()).dtype == torch.float16 else tensor_image.float()
    # Run model
    pred = model(tensor_image)
    pred = pred[0] if isinstance(pred, tuple) else pred
    return non_max_suppression_kpt(pred, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)


# Violence detection function
def run_vio_detection(img, model, device, half=True):  # Default to half precision
    img = img.to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference for violence detection
    with torch.no_grad():  # No gradient calculation
        with torch.cuda.amp.autocast(enabled=half):  # Automatic mixed precision
            pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, opt.iou_thres)  # set violence detection's confidence is 40%
    return pred


# Asynchronous detection function
def run_detection_async(source, camera_id, task_id, other_street):
    # Use global model & opt variables
    opt = global_opt
    obj_model = global_obj_model
    pose_model = global_pose_model
    vio_model = global_vio_model

    # Initialize variables
    device, weights, view_img, save_txt, imgsz = opt.device, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    half = True  # half precision always used
    cyan = [241, 218, 125]  # Color for 'person' class bounding box (cyan) / Hex : #7ddaf1

    stride = int(obj_model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['obj_model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = obj_model.module.names if hasattr(obj_model, 'module') else obj_model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    obj_model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(obj_model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:

        # Record TIME1
        t1 = time_synchronized()
        # Run Violence Detection
        print("Processing Violence Detection...")
        vio_img = torch.from_numpy(img).to(device)
        vio_pred = run_vio_detection(vio_img, vio_model, device, half=True)

        vio_results_str = ""
        level_sys = 0  # 0 is level 0, 1 is level 1, 2 is lvl2, 3 is lvl3; based on the conf.
        violence_count = 0
        for i, det in enumerate(vio_pred):  # detections per image
            if len(det):
                # Rescale boxes from model input size to original image size
                img_height, img_width = vio_img.shape[-2:]

                # for im0 in im0s: # for streaming source
                orig_height, orig_width = im0s.shape[:2]
                det[:, :4] = scale_coords((img_height, img_width), det[:, :4], (orig_height, orig_width)).round()

                # Filter and draw violence detections
                for *xyxy, conf, cls in reversed(det):
                    label_name = names[int(cls)]
                    conf_value = conf.item()  # Convert the 0-dim tensor to a number
                    # Level system based on the confidence
                    # if 0.55 <= conf_value < 0.58:
                    #     level_sys = 1
                    # elif 0.58 <= conf_value < 0.62:
                    #     level_sys = 2
                    # elif 0.62 <= conf_value:
                    #     level_sys = 3
                    if 0.25 <= conf_value < 0.35:
                        level_sys = 1
                    elif 0.35 <= conf_value < 0.65:
                        level_sys = 2
                    elif 0.65 <= conf_value:
                        level_sys = 3

                    if label_name == 'bicycle':  # Check if label is 'violence'; but why its bicycle for vio and person for non-vio...
                        violence_count += 1
                        label = f'Violence {conf:.2f}'
                        plot_one_box(xyxy, im0s, label=label, color=[0, 0, 255], line_thickness=2)
                        vio_results_str = f"Dangerous level {level_sys}, {violence_count} Violence, " # , {conf_value}
            else:
                vio_results_str = f"Dangerous level {level_sys}, 0 Violence, "

        # Resize image for pose estimation & Run pose estimation
        if not webcam:  # run pose estimation
            print("Processing Pose Estimation...")
            resized_im0s = letterbox(im0s, 960, stride=64, auto=True)[0]
            resized_tensor = transforms.ToTensor()(resized_im0s).unsqueeze(0).to(device)
            pose_pred = run_pose_estimation(resized_tensor, pose_model)
            pose_pred = output_to_keypoint(pose_pred)
        else:  # Else don't run pose estimation
            resized_im0s = im0s

        # Run Object Detection
        img = torch.from_numpy(img).to(device).half()  # convert to half
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]:
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                obj_model(img, augment=opt.augment)[0]

        # Inference
        with torch.no_grad():
            obj_pred = obj_model(img.half(), augment=opt.augment)[0]  # Make sure the input is in half precision

        # Apply NMS
        obj_pred = non_max_suppression(obj_pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            obj_pred = apply_classifier(obj_pred, modelc, img, resized_im0s)

        # Record TIME2
        t2 = time_synchronized()

        # Process drawing & saving imgs
        obj_results_str = ""
        for i, det in enumerate(obj_pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, resized_im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', resized_im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg

            # Print and draw object detections
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    color = cyan if names[int(cls)] == 'person' else colors[int(cls)]
                    plot_one_box(xyxy, im0, label=label, color=color, line_thickness=1)
                obj_results_str += f"{len(det)} {names[int(cls)]}{'s' * (len(det) > 1)}, "

            # Draw pose estimations on the image
            if not webcam:
                for idx in range(pose_pred.shape[0]):
                    plot_skeleton_kpts(im0, pose_pred[idx, 7:].T, 3)

            # Print time (inference + NMS)
            formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            global global_result  # changing global val
            with result_lock:  # changing result val within the lock
                global_result[task_id] = str(
                    f'Detected time: {formatted_time}. {vio_results_str}{obj_results_str}Done. ({(t2 - t1):.3f}s) Inference')
            print(global_result)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f"\nThe image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if not dataset.mode == 'image':  # if this is vid
                        if vid_writer is None or vid_path != path:  # if init new vid writer is needed
                            vid_path = path
                            if vid_writer is not None:
                                vid_writer.release()  # release the pre vid writer
                            fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if vid_cap else im0.shape[1]
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if vid_cap else im0.shape[0]
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                        # process each frame & write into video
                        if vid_writer is not None:
                            processed_im0 = cv2.resize(im0, (w, h))
                            vid_writer.write(processed_im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
        # make sure release the vid writer
    if vid_writer is not None:
        vid_writer.release()

    socketio.emit('ai_output', {'level': level_sys, 'index': int(camera_id), 'other_street': int(other_street)}, namespace='/')

    print(f'Detection done & Data sent to the server. ({time.time() - t0:.3f}s)')
    socketio.emit('detection_complete', {'task_id': task_id, 'result': global_result[task_id]})


@app.route('/run-detection', methods=['GET'])
def handle_run_detection():
    try:
        source = request.args.get('source', 'inference/images/ufc_sc_1.png')
        camera_id = request.args.get('camera_id', '0')
        other_street = request.args.get('other_street', '0')

        task_id = str(uuid.uuid4())

        # 初始化 global_results
        global_result[task_id] = "Detection in progress..."

        # 启动新线程进行检测
        threading.Thread(target=run_detection_async, args=(source, camera_id, task_id, other_street)).start()
    except Exception as e:
        global_result[task_id] = f"ERROR: {str(e)}"  # 更新结果为错误信息
        result = global_result.get(task_id, "No result available")
        return jsonify({"result": result})

    return jsonify({"message": "Detection started", "task_id": task_id})

@app.route('/reset-map', methods=['GET'])
def reset_map():
    socketio.emit('ai_output', {'level': int(0), 'index': int(0), 'other_street': int(4), 'map_reset': int(1)}, namespace='/')
    return jsonify({"message": "Map Reset Successfully"})

'''
@app.route('/run-detection', methods=['GET'])
def handle_run_detection():
    global global_result
    source = request.args.get('source', 'inference/images/ufc_sc_1.png')
    device = torch.device('cuda:0')  # Explicitly setting device to CUDA device 0
    camera_id = request.args.get('camera_id', '0')

    # Generate a new task ID
    task_id = str(uuid.uuid4())

    # Initialize the result entry in global_result
    global_result[task_id] = "Detection in progress..."

    # Start a new thread to handle detection
    detection_thread = threading.Thread(target=run_detection_async, args=(source, device, camera_id, task_id))
    detection_thread.start()

    # record start time & msg format
    t_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = (f"Detection started at " + str(t_start))

    # Wait for the thread finished
    detection_thread.join()

    # Build an array as a response
    response = [
        {"message": message},
        {"task_id": task_id},
        {"result": global_result[task_id]}
    ]

    return jsonify(response)
'''


if __name__ == '__main__':
    # Parse command line arguments and store in global variable opt
    opt = parse_args()

    # Start Flask application
    socketio.run(app, debug=True)
