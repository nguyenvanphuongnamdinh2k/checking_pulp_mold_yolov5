
import argparse
import os
import platform
import sys
from pathlib import Path

import torch


from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import threading
from threading import Thread
import datetime
from datetime import datetime, date
import os
import csv
import serial
from dataloaders import LoadStreams
class yolov5():
    def __init__(self):
        self.xop_bottom = []   # khi xuất hiện xốp dưới,số lượng của xốp dưới sẽ append vào mảng này ( if c == 0 : self.xop_bottom.append(int(n))  183,184
        self.bottom = []       # append len(self.xop_bottom) ==> kiểm tra số lượng xốp dưới đã được detect
        self.xop1 = []          # đếm số lượng của xốp dưới ( số lượng xốp dưới của 2 frame liên tiếp trừ đi nhau )
        self.xop_top = []  # tương tự 3 dòng trên (xốp dưới )
        self.top = []
        self.xop2 = []
        self.empty_box = []     # khi có hộp thì append giá trị 1 còn ko thì append giá trị 0
        self.TG = 0             # biến trung gian khởi tạo ban đầu
        self.box_rong = {}      # nếu ko có object trong 30 frame liên tiếp ==>self.box lên 1 ko thì vẫn bằng 0 ==> mục đích để reset lại quá trình check
        self.box = 0     # cái trên đã giải thích
        self.printer = 0  # khong co may in: printer = 0 else printer = 1
        self.checkduoi = 0          # kiểm tra xem check 4 dưới đã oki chưa ,nếu oki thì lên 1
        self.timestamp1 = 0         # thời gian để reset tín hiệu sau 2 giây và để hiển thị tín hiệu done
        #############################################
        self.cover = 0                # cover trường hợp bắt thùng xốp thành máy in ( xảy ra vấn đề sẽ hiển thị thiếu 4 con dưới)           # phuong add


        # self.port = "COM5"  # Thay thế bằng số cổng Serial mà Hercules Serial đang sử dụng trên máy tính của bạn        # phuong add
        # self.baudrate = 57600                                                                                           # phuong add
        # self.ser = serial.Serial(self.port, self.baudrate)                                                              # phuong add
        # self.hex_data =None
        self.create_csv()  # phuong add

        self.dem_duoi = 0
        self.dem_tren = 0
        self.TG_duoi = 0
        self.TG_tren = 0

    def create_csv(self):  # tạo file csv để lưu kết quả và thời gian
        self.date = date.today()  # năm -tháng - ngày hiện tại được lấy trong máy
        if os.path.isfile(os.path.join("file_csv", str(self.date) + '.csv')):  # kiểm tra file tồn tại
            print('file exists')
        else:
            self.title = ['bottom', 'top']  # tạo title
            with open(os.path.join("file_csv", str(self.date) + '.csv'), 'a+', newline='') as file:
                writer = csv.writer(file)  # lưu file
                writer.writerow(self.title)  #
                file.close()

    def run2(self,
             weights=r'best__15.pt',  # file weight
             # source=r"rtsp://admin:admin111@192.168.1.160:554/cam/realmonitor?channel=1&qsubtype=00",
             # file/dir/URL/glob/screen/0(webcam)
             source = r"alo.mp4",
             data='data/coco128.yaml',  # dataset.yaml path
             imgsz=(640, 640),  # inference size (height, width)
             conf_thres=0.7,  # confidence threshold
             iou_thres=0.45,  # NMS IOU threshold
             max_det=1000,  # maximum detections per image
             device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
             view_img=True,  # show results
             save_txt=False,  # save results to *.txt
             save_conf=False,  # save confidences in --save-txt labels
             save_crop=False,  # save cropped prediction boxes
             nosave=False,  # do not save images/videos
             classes=None,  # filter by class: --class 0, or --class 0 2 3
             agnostic_nms=False,  # class-agnostic NMS
             augment=False,  # augmented inference
             visualize=False,  # visualize features
             update=False,  # update all models
             project='runs/detect',  # save results to project/name
             name='exp',  # save results to project/name
             exist_ok=False,  # existing project/name ok, do not increment
             line_thickness=2,  # bounding box thickness (pixels)
             hide_labels=False,  # hide labels
             hide_conf=False,  # hide confidences
             half=False,  # use FP16 half-precision inference
             dnn=False,  # use OpenCV DNN for ONNX inference
             vid_stride=1,  # video frame-rate stride
             ):
        # lưu file
        self.source = source
        save_img = not nosave and not self.source.endswith('.txt')  # save inference images ,có lưu ko
        is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # kiểm tra là file ko
        is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = self.source.isnumeric() or self.source.endswith('.streams') or (
                    is_url and not is_file)  # nếu là webcam thì đầu vào là số hoặc đuôi là .stream hoặc là url
        screenshot = self.source.lower().startswith('screen')
        if is_url and is_file:
            self.source = check_file(self.source)  # download           # đường dẫn trên mạng

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir  ,tạo chỗ để lưu

        # Load model
        device = select_device(device)  # chọn CPU hoặc GPU
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # load weight
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloaderq
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(self.source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(self.source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            # a = 0
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    # s += f'{i}: '
                    s += ''
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                # s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, 5].unique():
                        a = 0  # ko có object trong khung hình
                        self.printer = 0
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            print(xywh)
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if c == 0:
                                self.xop_bottom.append(int(n))
                            if c == 1:
                                self.printer = 1
                            if c == 3:
                                self.xop_top.append(int(n))
                            if c == 2:
                                a = 1
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=False)

                # ---------------------------------Start signal---------------------------------------------------------
                self.timestamp = datetime.timestamp(datetime.now())

                now = datetime.now()
                hour = now.hour
                minute = now.minute
                second = now.second

                if len(det) > 0:
                    self.empty_box.append(1)
                if len(det) == 0:
                    self.empty_box.append(0)

                if len(self.empty_box) > 15:
                    self.box_rong = set(self.empty_box[-15:])
                # file = open(str(self.date) + '.csv', 'a+', newline='')
                self.timestamp = datetime.timestamp(datetime.now())
                '''
                signal start
                '''
                if (self.printer == 0 and len(self.xop_bottom) != 0) or self.TG == 0:
                    self.bottom.append(len(self.xop_bottom))
                    if len(self.bottom) >= 2:
                        x1 = self.bottom[-1]
                        x2 = self.bottom[-2]
                        self.xop1.append(x1 - x2)
                '''
                signal stop
                '''
                if self.printer == 1 and self.TG == 0 and a == 1:  # phuong add
                    self.cover = 1  # phuong add
                if self.cover == 1:  # phuong add
                    # check kq 4 bottom, if OK ==> count top , else : pack more pulp mold
                    if len(self.xop1) != 0 and max(self.xop1) >= 4:
                        self.TG = 1
                        self.checkduoi = 1
                        self.xop_bottom = []
                        self.cover = 0  # phuong add
                        # self.hex_data = "02 F3 01 01 F5 03"  # phuong add
                        # self.byte_data = bytes.fromhex(self.hex_data)  # phuong add
                        # self.ser.write(self.byte_data)  # phuong add
                    if len(self.xop1) == 0 or max(self.xop1) < 4:
                        cv2.putText(im0, 'duoi: dong them pulp mold', (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                    (0, 0, 255), 3)

                        self.TG = 0
                        # self.hex_data = "02 F3 01 00 F4 03"  # phuong add
                        # self.byte_data = bytes.fromhex(self.hex_data)  # phuong add
                        # self.ser.write(self.byte_data)  # phuong add
                        self.TG_duoi += 1
                        if self.TG_duoi == 1:
                            self.dem_duoi += 1
                            with open(os.path.join("file_csv", str(self.date) + '.csv'), 'a+', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([f"{hour}:{minute}:{second}", ""])
                if self.TG == 1:
                    cv2.putText(im0, 'duoi: OK', (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
                    if len(self.xop_top) != 0:
                        self.top.append(len(self.xop_top))
                        if len(self.top) >= 2:
                            x1 = self.top[-1]
                            x2 = self.top[-2]
                            self.xop2.append(x1 - x2)
                '''
                Reset kq
                '''
                if self.checkduoi == 1:
                    if self.box_rong == {0}:
                        self.box = 1

                if self.box == 1 and self.TG != 3:

                    if len(self.xop2) != 0 and max(self.xop2) >= 4:
                        cv2.putText(im0, 'tren: OK', (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
                        self.TG = 2
                        # self.hex_data = "02 F3 01 01 F5 03"  # phuong add
                        # self.byte_data = bytes.fromhex(self.hex_data)  # phuong add
                        # self.ser.write(self.byte_data)  # phuong add

                    if (len(self.xop2) != 0 and max(self.xop2) < 4) or len(self.xop2) == 0:
                        cv2.putText(im0, 'tren: dong them pulp mold', (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                    (0, 0, 255), 3)
                        self.TG = 1
                        # self.hex_data = "02 F3 01 00 F4 03"  # phuong add
                        # self.byte_data = bytes.fromhex(self.hex_data)  # phuong add
                        # self.ser.write(self.byte_data)  # phuong add
                        self.TG_tren += 1
                        if self.TG_tren == 1:
                            self.dem_tren += 1
                            with open(os.path.join("file_csv", str(self.date) + '.csv'), 'a+', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(["", f"{hour}:{minute}:{second}"])
                if self.TG == 2:
                    self.timestamp1 = datetime.timestamp(datetime.now())
                    self.TG = 3
                if self.timestamp1 < self.timestamp < self.timestamp1 + 5 and self.timestamp1 != 0:
                    # print('aaaa')
                    cv2.putText(im0, 'Done', (60, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)

                if self.timestamp > self.timestamp1 + 2 and self.timestamp1 != 0 and self.TG == 3 and self.box_rong == {
                    0}:
                    self.xop_bottom = []
                    self.bottom = []
                    self.xop1 = []
                    self.xop_top = []
                    self.top = []
                    self.xop2 = []
                    self.TG = 0
                    self.xop2 = []
                    self.box_rong = {}
                    self.box = 0
                    self.printer = 0
                    self.checkduoi = 0
                    self.cover = 0  # phuong add
                    self.hex_data = None  # phuong add
                    self.TG_duoi = 0
                    self.TG_tren = 0
                # ---------------------------------Start signal---------------------------------------------------------

                # Stream results
                im0 = annotator.result()
                # print(self.printer)

                im_copy = im0.copy()
                im_copy = im_copy[460:910, 620:1240]
                if view_img:
                    cv2.putText(im0, str(datetime.now().strftime("%H:%M:%S")), (1330, 90), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                (0, 0, 255), 3)
                    cv2.putText(im0, "duoi:" + str(self.dem_duoi), (610, 750), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                (0, 255, 255), 2)
                    cv2.putText(im0, "tren:" + str(self.dem_tren), (610, 850), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                (0, 255, 255), 2)
                    cv2.imshow('2a', im0)
                    cv2.imshow('aaaaa', im_copy)
                    cv2.waitKey(1)  # 1 millisecondq

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)


if __name__ == '__main__':
    main_my = yolov5()
    main_my.run2()





