import numpy as np
import os.path as osp
import cPickle
import cv2
from glob import glob
from argparse import ArgumentParser
from scipy.misc import imsave

import caffe


class ObjLocPreprocessor(object):
    def __init__(self, crop_height, crop_width, center_only):
        super(ObjLocPreprocessor, self).__init__()
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.center_only = center_only


    def pre_process(self, original, resized):
        # All the bboxes are in the format of: (x1, y1, x2, y2)
        inputs = []
        self.cropped_bboxes = []
        self.original_shapes = []
        self.resized_shapes = []
        # Center crop / multi-crop
        if self.center_only:
            for ori, res in zip(original, resized):
                center = np.asarray(res.shape[:2]) // 2
                crop = [center[1] - self.crop_width // 2,
                        center[0] - self.crop_height // 2]
                crop.append(crop[0] + self.crop_width)
                crop.append(crop[1] + self.crop_height)
                inputs.append(res[crop[1]:crop[3], crop[0]:crop[2], :])
                self.cropped_bboxes.append(crop)
                self.original_shapes.append(ori.shape[:2])
                self.resized_shapes.append(res.shape[:2])
        else:
            for ori, res in zip(original, resized):
                # center crop
                center = np.asarray(res.shape[:2]) // 2
                crop = [center[1] - self.crop_width // 2,
                        center[0] - self.crop_height // 2]
                crop.append(crop[0] + self.crop_width)
                crop.append(crop[1] + self.crop_height)
                inputs.append(res[crop[1]:crop[3], crop[0]:crop[2], :])
                self.cropped_bboxes.append(crop)
                # 4 half-corner crops
                crop_x = [(res.shape[1] - self.crop_width) // 4,
                          (res.shape[1] - self.crop_width) * 3 // 4]
                crop_y = [(res.shape[0] - self.crop_height) // 4,
                          (res.shape[0] - self.crop_height) * 3 // 4]
                for x in crop_x:
                    for y in crop_y:
                        crop = [x, y, x + self.crop_width, y + self.crop_height]
                        inputs.append(res[crop[1]:crop[3], crop[0]:crop[2], :])
                        self.cropped_bboxes.append(crop)
                # mirrors
                for k in xrange(5):
                    inputs.append(inputs[-5][:, ::-1, :])
                    crop = self.cropped_bboxes[-5]
                    self.cropped_bboxes.append(
                        [crop[2], crop[1], crop[0], crop[3]])
                # record duplicate shapes
                self.original_shapes.extend([ori.shape[:2]] * 10)
                self.resized_shapes.extend([res.shape[:2]] * 10)
        self.cropped_bboxes = np.asarray(self.cropped_bboxes)
        self.original_shapes = np.asarray(self.original_shapes)
        self.resized_shapes = np.asarray(self.resized_shapes)
        return np.asarray(inputs)


    def post_process(self, loc_ret):
        # Will provide (x1, y1, x2, y2) relative to the original image size
        num_classes = 1000
        loc_ret = loc_ret.reshape(loc_ret.shape[0], num_classes, 4)
        n = min(loc_ret.shape[0], self.cropped_bboxes.shape[0])
        ret = np.zeros_like(loc_ret)
        for image_id in xrange(n):
            original_height, original_width = self.original_shapes[image_id]
            resized_height, resized_width = self.resized_shapes[image_id]
            x1, y1, x2, y2 = self.cropped_bboxes[image_id]
            # map from loc. result to cropped image's coor.
            x = (loc_ret[image_id, :, 0] * (x2 - x1) + (x1 + x2) / 2.0).round().astype(np.int32)
            y = (loc_ret[image_id, :, 1] * (y2 - y1) + (y1 + y2) / 2.0).round().astype(np.int32)
            w = (np.exp(loc_ret[image_id, :, 2]) * abs(x2 - x1)).round().astype(np.int32)
            h = (np.exp(loc_ret[image_id, :, 3]) * abs(y2 - y1)).round().astype(np.int32)
            # map from (xc, yc, w, h) to (x1, y1, x2, y2)
            ret[image_id, :, 0] = x - w // 2
            ret[image_id, :, 1] = y - h // 2
            ret[image_id, :, 2] = ret[image_id, :, 0] + w
            ret[image_id, :, 3] = ret[image_id, :, 1] + h
            ret[image_id, :, 0] = np.maximum(ret[image_id, :, 0], 0) * original_width // resized_width
            ret[image_id, :, 1] = np.maximum(ret[image_id, :, 1], 0) * original_height // resized_height
            ret[image_id, :, 2] = np.minimum(ret[image_id, :, 2], resized_width) * original_width // resized_width
            ret[image_id, :, 3] = np.minimum(ret[image_id, :, 3], resized_height) * original_height // resized_height
        return ret


def do_localization(net, transformer, inputs):
    caffe_in = np.zeros(np.asarray(inputs.shape)[[0, 3, 1, 2]],
                        dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        caffe_in[ix] = transformer.preprocess('data', in_)
    out = net.forward(blobs=['fc8-loc'], **{'data': caffe_in})
    return out['fc8-loc']


def main(args):
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()

    # Preset multi-view test parameters
    mean = np.asarray([104, 117, 123])
    batch_size = 50
    num_gpus = 2 # will split files to be processed according to this

    # Init net and transformer
    net = caffe.Net(args.model_def, args.pretrained_model, caffe.TEST)

    objloc_prep = ObjLocPreprocessor(net.blobs['data'].data.shape[2],
                                     net.blobs['data'].data.shape[3],
                                     args.center_only)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', mean)

    # Get input files list
    original_folder = osp.expanduser(args.original_image_folder)
    resized_folder = osp.expanduser(args.resized_image_folder)
    original_files = sorted(glob(osp.join(original_folder, '*.JPEG')))
    resized_files = sorted(glob(osp.join(resized_folder, '*.JPEG')))

    num_files = len(original_files)

    if args.center_only:
        ret = np.zeros((num_files, 1, 1000, 4), dtype=np.float32)
        num_batches = (num_files - 1) // batch_size + 1
        for batch_id in xrange(num_batches):
            print "Batch {} / {}".format(batch_id, num_batches)
            begin = batch_id * batch_size
            end = min(begin + batch_size, num_files)
            original_images = [cv2.imread(original_files[i], cv2.CV_LOAD_IMAGE_COLOR)
                               for i in xrange(begin, end)]
            resized_images = [cv2.imread(resized_files[i], cv2.CV_LOAD_IMAGE_COLOR)
                              for i in xrange(begin, end)]
            inputs = objloc_prep.pre_process(original_images, resized_images)
            outputs = do_localization(net, transformer, inputs)
            outputs = objloc_prep.post_process(outputs)
            ret[begin:end] = outputs.reshape(outputs.shape[0], 1, 1000, 4)
    else:
        # 5 crops x mirror / no mirror
        num_files_per_gpu = (num_files - 1) // num_gpus + 1
        files_begin = args.gpu * num_files_per_gpu
        files_end = min(files_begin + num_files_per_gpu, num_files)
        original_files = original_files[files_begin:files_end]
        resized_files = resized_files[files_begin:files_end]
        ret = np.zeros((num_files_per_gpu, 10, 1000, 4), dtype=np.float32)
        num_batches = (num_files_per_gpu * 10 - 1) // batch_size + 1
        for batch_id in xrange(num_batches):
            print "Batch {} / {}".format(batch_id, num_batches)
            begin = batch_id * batch_size // 10
            end = min((batch_id + 1) * batch_size // 10, num_files_per_gpu)
            original_images = [cv2.imread(original_files[i], cv2.CV_LOAD_IMAGE_COLOR)
                               for i in xrange(begin, end)]
            resized_images = [cv2.imread(resized_files[i], cv2.CV_LOAD_IMAGE_COLOR)
                              for i in xrange(begin, end)]
            inputs = objloc_prep.pre_process(original_images, resized_images)
            outputs = do_localization(net, transformer, inputs)
            outputs = objloc_prep.post_process(outputs)
            ret[begin:end] = outputs.reshape(end - begin, 10, 1000, 4)
    np.save(args.output_file, ret)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('original_image_folder')
    parser.add_argument('resized_image_folder')
    parser.add_argument('output_file')
    parser.add_argument('--model_def', required=True)
    parser.add_argument('--pretrained_model', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--center_only', action='store_true')
    args = parser.parse_args()
    main(args)