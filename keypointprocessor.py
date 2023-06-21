import base64
import io
import time

import openpifpaf
import PIL
import torch


class Processor(object):
    def __init__(self, width_height, args):
        self.width_height = width_height

        # Loading model
        self.model, _ = openpifpaf.network.nets.factory_from_args(args) #initialize neural network model
        self.model = self.model.to(args.device) #model to device(cpu or gpu)
        self.processor = openpifpaf.decoder.factory_from_args(args, self.model) # initializes decoder (component processesing output of the neural network to obtain the desired output for the task)
        self.device = args.device

#akes a base64-encoded image string as input, 
# preprocesses it, runs it through a computer vision model, 
# returns the resulting keypoint sets and scores.
    def single_image(self, b64image):
        image_bytes = io.BytesIO(base64.b64decode(b64image))
        im = PIL.Image.open(image_bytes).convert('RGB')

        #image resized if needed
        target_wh = self.width_height
        if (im.size[0] > im.size[1]) != (target_wh[0] > target_wh[1]):
            target_wh = (target_wh[1], target_wh[0])
        if im.size[0] != target_wh[0] or im.size[1] != target_wh[1]:
            print(f'!!! have to resize image to {target_wh} from {im.size}')
            im = im.resize(target_wh, PIL.Image.BICUBIC) #using cubic splines or other polynomial technique for sharpening and enlarging the images
        width_height = im.size

        #preprocessing
        start = time.time()
        preprocess = openpifpaf.transforms.EVAL_TRANSFORM
        processed_image_cpu, _, __ = preprocess(im, [], None)
        processed_image = processed_image_cpu.contiguous().to(self.device, non_blocking=True)
        print(f'preprocessing time {time.time() - start}')

        all_fields = self.processor.fields(torch.unsqueeze(processed_image.float(), 0))[0]
        keypoint_sets, scores = self.processor.keypoint_sets(all_fields) # extract the keypoint sets and scores from the output fields

        # Normalize scale
        keypoint_sets[:, :, 0] /= processed_image_cpu.shape[2]
        keypoint_sets[:, :, 1] /= processed_image_cpu.shape[1]

        return keypoint_sets, scores, width_height