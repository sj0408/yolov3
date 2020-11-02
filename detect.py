import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


def detect(save_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt, conf_thres, iou_thres = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt, opt.conf_thres, opt.iou_thres
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11)

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img)[0].float() if half else model(img)[0]
        t2 = torch_utils.time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        if pred[0] is None:
            pred[0] = torch.empty((0,6)).to(device)
            
        sub_source = os.path.splitext(path.split('/')[-1])[0]
        slice_dataset = LoadImages(f'../test/sliced/images/{sub_source}', img_size=416)
       
        s_preds = torch.empty(0,6).to(device)
        
        for s_path, s_img, s_im0s, s_vid_cap in slice_dataset:

            s_img = torch.from_numpy(s_img).to(device)
            s_img = s_img.half() if half else s_img.float()  # uint8 to fp16/32
            s_img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if s_img.ndimension() == 3:
                s_img = s_img.unsqueeze(0)

            # Inference sliced image and stich result
            s_pred = model(s_img)[0].float() if half else model(s_img)[0]

            s_pred = non_max_suppression(s_pred, conf_thres=conf_thres, iou_thres=iou_thres)

            if s_pred[0] is not None:
                grid_num = int(os.path.splitext(s_path.split('/')[-1])[0].split('_')[-1])

                # stride values
                w_stride = (grid_num-1)%3
                h_stride = (grid_num-1)//3

                # coordinate adjust for stiching
                s_pred[0][:,0] += (img_size * 0.75 * w_stride)
                s_pred[0][:,2] += (img_size * 0.75 * w_stride)
                s_pred[0][:,1] += (img_size * 0.75 * h_stride)
                s_pred[0][:,3] += (img_size * 0.75 * h_stride)

                s_preds = torch.cat((s_preds, s_pred[0]))

        stiched_output = non_max_suppression_ver_2([s_preds], conf_thres=conf_thres, iou_thres=iou_thres)

        # coordinate rescale to 416 X 416
        if stiched_output[0] is None:
            stiched_output[0] = torch.empty((0,6)).to(device)
        else:
            w_scale = img_size / 1040   # need to be in variable
            h_scale = img_size / 728    # need to be in variable

            stiched_output[0][:,0] *= w_scale
            stiched_output[0][:,2] *= w_scale
            stiched_output[0][:,1] *= h_scale
            stiched_output[0][:,3] *= h_scale    
        
        # concatnate output
        cat_output = torch.cat((pred[0], stiched_output[0]))
        final_output = [cat_output]

        # NMS for original image and stiched image
        final_output = non_max_suppression_ver_2(final_output, conf_thres=0.2, iou_thres=0.4)
        
        # Process detections
        for i, det in enumerate(final_output):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
