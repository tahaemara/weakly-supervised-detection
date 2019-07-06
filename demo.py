"""
Created on Fri Jul 3 18:12:30 2019

@author: Taha Emara  @email: taha@emaraic.com

"""
import PIL
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, models
import cv2
import argparse

CLasses = {0: 'Bicycle', 1: 'Car', 2: 'Motorbike'}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
cv2.namedWindow('Results', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Results', 600, 400)

extracted_features = []


def get_feature(module, input, output):
    extracted_features.append(output.data.cpu().numpy())


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=False,
                    help='path to pretrained model', default='experiments/experiment_2019-07-03_13_18/epoch 28.pth')
    ap.add_argument('--image_path', required=False,
                    help='path to input image',
                    default='test_samples/2128649873.jpg') #test_samples/41992186.jpg , test_samples/39392419.jpg
    ap.add_argument('--numclasses', required=False,
                    help='number of classes', default=3)
    args = ap.parse_args()
    return args


def load_model(model_path, numclasses):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, numclasses)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    return model


def load_image(image_path):
    img = PIL.Image.open(image_path)
    opencv_image = np.array(img)

    # Convert RGB to BGR
    opencv_image = opencv_image[:, :, ::-1].copy()
    input1 = data_transforms['val'](img)
    img2 = input1.unsqueeze(0)
    image = Variable(img2).to(device)
    return image, opencv_image


def put_text(image, text, boundRect):
    cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)

    draw = ImageDraw.Draw(pil_im)

    font = ImageFont.load_default()  # truetype("/Library/Fonts/Arial Black.ttf", 40)

    draw.text((int(boundRect[0] + boundRect[2] / 2) - 50, int(boundRect[1] + boundRect[3] / 2) - 50), text, font=font,
              fill=(255, 255, 255))

    image = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    return image


def get_cam(model, weight_softmax, image, opencv_image):
    image_w, image_h = opencv_image.shape[1], opencv_image.shape[0]
    input_img = opencv_image.copy()
    with torch.no_grad():
        outputs = model.forward(image)
        predictions = torch.max(outputs, 1)[1]
        class_id = predictions.item()

        print(class_id, "   ", CLasses[class_id])

        bz, nc, h, w = extracted_features[0].shape

        cam = weight_softmax[class_id].dot(extracted_features[0].reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        cam_img = cv2.resize(cam_img, (image_w, image_h))
        heat_map = cam_img.copy()

        ret, cam_img = cv2.threshold(cam_img, 170, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(cam_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            boundRect = cv2.boundingRect(contour)
            cv2.drawContours(opencv_image, contours, i, (255, 255, 0, 0), 3, 4, hierarchy)
            cv2.rectangle(opencv_image, (int(boundRect[0]), int(boundRect[1])), \
                          (int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3])), (255, 0, 0, 0), 2)
            opencv_image = put_text(opencv_image, CLasses[class_id], boundRect)
            # cv2.rectangle(cam_img, (int(boundRect[0]), int(boundRect[1])), \
            #               (int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3],3)), (255, 0, 0), 2)
        # cv2.imwrite('Original ' + str(class_id) + "_" + CLasses[class_id] + '.jpg', opencv_image)
        # cv2.imwrite('CAM ' + str(class_id) + "_" + CLasses[class_id] + '.jpg', cam_img)
        image_to_show = np.hstack((input_img, cv2.cvtColor(heat_map, cv2.COLOR_GRAY2BGR),
                                   cv2.cvtColor(cam_img, cv2.COLOR_GRAY2BGR), opencv_image))
        cv2.imwrite('im.jpg', image_to_show)
        cv2.imshow('Results', image_to_show)
        cv2.waitKey()


def main(args):
    model = load_model(args.model_path, args.numclasses)
    model.eval()

    model.layer4.register_forward_hook(get_feature)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    image, opencv_image = load_image(args.image_path)
    get_cam(model, weight_softmax, image, opencv_image)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
    cv2.destroyAllWindows()
