import numpy as np
import tensorflow as tf
import cv2
import imutils
from time import time
import matplotlib.pyplot as plt
from datetime import datetime
import random

ABE_PATH = "./Images/abe.png"
EARTH_PATH = "./Images/earth_1024.jpg"

# INITILAIAZATION_METHOD = 'Naive'
WEIGHTS_STDEV = 0.01
BIAS_CONST = 0.01
INITILAIAZATION_METHOD = 'Xavier'
CONV_KERNEL_SIZE = 3
INPUT_CHANNELS = 1
CONV_CHANNELS = 64
CONV_LAYERS_NUM = 8
ITERATIONS = 200
INTER_SCALES_NUM = 6
# MAX_DOWNSCALE_FATHERS = 0.2
MAX_DOWNSCALE_FATHERS = 0.4
DOWNSCALE_FATHERS_NUM = 6
# CROP_SIZE = 128
CROP_SIZE = 64

#IMAGE PROCESSING
DOWN_SAMPLING_METHOD = cv2.INTER_AREA
UP_SAMPLING_METHOD = cv2.INTER_LINEAR

#TODO
#please check that the downscaled images are larger then 128x128

def get_logdir():
    # Return unique logdir based on datetime
    now = datetime.utcnow().strftime("%m%d%H%M%S")
    logdir = "run-{}".format(now)

    return logdir

def load_image(path, verbose = False):
    # loads RGB image as float32 and outputs the luminance channel
    #TODO - convert to luminance channel, test it!
    img = cv2.imread(path)
    # img = cv2.imread(path).astype('float32')/255.
    #img_HSV = np.float32(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    img_avg = np.mean(img, axis=2).astype("float32")/255.0 #this is the mean of the RGB channels
    if verbose:
        #test
        cv2.imshow("Original image", img)
        cv2.waitKey(0)
        cv2.imshow("The image lightness (computed as a mean)", img_avg)
        cv2.waitKey(0)
        #image = np.array(Image.open(os.path.join(path, file)))
    return img_avg

def rgb_mean_normalization(image):
    # is neccesary?
    #if so, should denormalize output later
    pass

def rotations_and_flips(image):
    # performs D_4 (8) rotations and flips on image. returns list of images
    # TODO: deal with multiplue images of different sizes

    image_c = image.copy()
    images = [image_c, image_c[::-1, :]]

    image_90 = image_c.transpose([1, 0])[::-1, :]
    images += [image_90, image_90[::-1, :]]

    image_180 = image_c[::-1, ::-1]
    images += [image_180, image_180[::-1, :]]

    image_270 = image_c.transpose([1, 0])[:, ::-1]
    images += [image_270, image_270[::-1, :]]

    ''' An old version for 3-D arrays
    image_c = image.copy()
    images = [image_c, image_c[::-1, :, :]]

    image_90 = image_c.transpose([1, 0, 2])[::-1, :, :]
    images.append(image_90)
    images.append(image_90[::-1, :, :])

    image_180 = image_c[::-1, ::-1, :]
    images.append(image_180)
    images.append(image_180[::-1, :, :])

    image_270 = image_c.transpose([1, 0, 2])[:, ::-1, :]
    images.append(image_270)
    images.append(image_270[::-1, :, :])'''

    return images


def rand_crop_fathers_sons(HR_fathers_list, LR_sons_list, crop_size):
    # randomly crops images patch of size shape
    # corresponding images in HR and LR lists are of the same size!
    N = len(HR_fathers_list)
    # test
    if len(LR_sons_list) != N:
        print "rand_crop_fathers_sons function: ERROR, LR sons and HR fathers lists are of different lengths"

    print "Random cropping started..."
    HR_fathers_crp= []
    LR_sons_crp = []

    for i in range(N):
        HR_image = HR_fathers_list[i]
        LR_image = LR_sons_list[i]
        # test
        if HR_image.shape != LR_image.shape:
            print "rand_crop_fathers_sons function: ERROR, LR and HR images are of different shapes"
        row_crop_start = random.randint(0, HR_image.shape[0] - crop_size)
        col_crop_start = random.randint(0, HR_image.shape[1] - crop_size)
        HR_fathers_crp.append(HR_image[row_crop_start:row_crop_start + crop_size, col_crop_start:col_crop_start + crop_size])
        LR_sons_crp.append(LR_image[row_crop_start:row_crop_start + crop_size, col_crop_start:col_crop_start + crop_size])

    print "Random cropping ended..."
    return HR_fathers_crp, LR_sons_crp

def downscale_to_size(images, scale):
    # performs interpolation on images list by scale
    # Preferable interpolation methods are cv2.INTER_AREA for shrinking and cv2.INTER_CUBIC(slow) & cv2.INTER_LINEAR for zooming
    #TODO: change the interpolation method to bicubic with antialiasing
    dwn_resized_images = []
    print "Down scaling by factor {} began...".format(scale)
    for i in range(len(images)):
        dwn_resized_images.append(cv2.resize(images[i],None, fx=scale, fy=scale, interpolation=DOWN_SAMPLING_METHOD))
    print "Down scaling by factor {} ended.".format(scale)
    return dwn_resized_images

def upscale_to_size(images, scale):
    # Preferable interpolation methods are cv2.INTER_AREA for shrinking and cv2.INTER_CUBIC(slow) & cv2.INTER_LINEAR for zooming
    # TODO: change the interpolation method to bicubic with antialiasing
    up_resized_images = []
    print "Down scaling by factor {} began...".format(scale)
    for i in range(len(images)):
        up_resized_images.append(cv2.resize(images[i], None, fx=scale, fy=scale, interpolation=UP_SAMPLING_METHOD))
    print "Down scaling by factor {} ended.".format(scale)
    return up_resized_images

def create_HR_fathers(I, down_scale_list):
    # creates set of HR fathers with scales sampling_scale_list, and perform D_4 on all fathers
    # The down_scale_list should not include the scaling factor 1
    HR_fathers = [] + rotations_and_flips(I)
    for s in down_scale_list:
        scaled_img = downscale_to_size([I], s)
        HR_fathers += rotations_and_flips(scaled_img[0])
    return HR_fathers

def rotate_flip_set(image_set):
    # perform D_4 actions on the images set
    pass

def create_LR_sons(HR_set, scale, verbose=False):
    # create set of LR sons from HR_set according to scale (scale < 1)
    ''' OLD CODE
    LR_sons = downscale_to_size(HR_set, scale)'''
    LR_sons = []
    if verbose:
        print "Creating LR sons according to facor {} began...".format(scale)
    for i in range(len(HR_set)):
        # dwn_scale_shape = int(HR_set[i].shape * scale)
        # dwn_scale_img = cv2.resize(HR_set[i], dwn_scale_shape, interpolation=DOWN_SAMPLING_METHOD)
        dwn_scale_img = cv2.resize(HR_set[i], None, fx=scale, fy=scale, interpolation=DOWN_SAMPLING_METHOD) #another implementation
        LR_sons.append(cv2.resize(dwn_scale_img, HR_set[i].shape[::-1], interpolation=UP_SAMPLING_METHOD))
    if verbose:
        print "Creating LR sons according to facor {} ended.".format(scale)
    return LR_sons

def weight_variable(shape):
    # weight definition and initialization
    if INITILAIAZATION_METHOD == 'Xavier':
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        W = tf.Variable(initializer(shape))

    else:
        # Naive initilazation
        initial = tf.truncated_normal(shape, stddev=WEIGHTS_STDEV)
        W = tf.Variable(initial)
    return W


def bias_variable(shape):
    # bias definition and initialization
    if INITILAIAZATION_METHOD == 'Xavier':
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        b = tf.Variable(initializer([shape]))
    else:
        # Naive initiliaztion
        initial = tf.constant(BIAS_CONST, shape=[shape])
        b = tf.Variable(initial)
    return b


def conv2d(x, W, stride=1, pad = 'SAME'):
    # basic conv

    # x should be: [batch, in_height, in_width, in_channels]
    # W is of shape [filter_height, filter_width, in_channels, out_channels]

    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=pad)

def conv_layer(input_tensor, shape, stride=1, pad = 'SAME', act=tf.nn.relu, name = 'conv'):
    # conv layer wrapper

    # input_tensor should be: [batch, in_height, in_width, in_channels]
    # shape is [filter_height, filter_width, in_channels, out_channels]

    with tf.name_scope(name):
        with tf.name_scope('weights'):
            weights = weight_variable(shape)
        with tf.name_scope('biases'):
            biases = bias_variable(shape[-1])
        with tf.name_scope('Wx_plus_b'):
            preactivate = conv2d(input_tensor, weights, stride, pad) + biases
            tf.summary.histogram('pre_activations', preactivate)
        if act:
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations
        else:
            return preactivate


def L1_loss(HR_prediction, HR_truth):
    # return L1 loss
    return tf.reduce_mean(tf.abs(HR_prediction - HR_truth))

def forward_prop(LR):
    # perform forward propogation on LR HR pairs, returns output images
    # input is already cropped and interpolated to output size (I^s) (one tensor for LR and one for HR)

    # LR should be: [batch, in_height, in_width, in_channels]
    # conv shape is [filter_height, filter_width, in_channels, out_channels]

    # initial convolution on input:
    conv_activation = conv_layer(LR, [CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, INPUT_CHANNELS, CONV_CHANNELS], stride=1, act=tf.nn.relu, name='Input_Conv')

    # remaining CONV_LAYERS_NUM-1 layers:
    for i in range(1, CONV_LAYERS_NUM):
        layer_name = 'Hidden_Conv{}'.format(i) # 2,3,4,...,CONV_LAYERS_NUM-1
        conv_activation = conv_layer(conv_activation, [CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV_CHANNELS, CONV_CHANNELS], stride=1, act=tf.nn.relu, name=layer_name)

    # output layer, back to 3 channels:
    output = conv_layer(conv_activation, [CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV_CHANNELS, INPUT_CHANNELS], stride=1, act=None, name='Output_Conv')
    # output = conv_layer(conv_activation, [CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV_CHANNELS, INPUT_CHANNELS], stride=1, act=tf.nn.relu, name='Output_Conv')

    return output

def produce_test_image(I, scale):
    # interpolate image to scale
    # create d_4 group
    # forward_prop each member of group
    # unify outputs - rotate and flip back, perform median (separate function?)
    I_upscaled = cv2.resize(I, None, fx=scale, fy=scale, interpolation=UP_SAMPLING_METHOD)
    I_upscaled = np.expand_dims(I_upscaled,axis=0)
    I_upscaled = np.expand_dims(I_upscaled,axis=3)
    # I_upscaled = upscale_to_size(I, scale)

    I_SR = forward_prop(I_upscaled)

    return I_SR[0,:,:,0]

    # I_rotated_flipped = rotations_and_flips(I_upscaled)
    # I_forward_proped = []
    # for single_image in I_rotated_flipped:
    #     I_forward_proped.append(forward_prop(single_image))
    # I_forward_proped list is ordered the same as I_rotated_flipped, so can rotate and flip back accordingly
    # perform median/back-projection of the 8 images
    # return merged image

def ZSSR(image_path, desired_scale, verbose = False):
    # perform entire pipeline on single image

    # load image
    # define scale_list for father creation
    # create session, initialize variables
    # create training set - fathers and sons, downsizing by scale, upscaling back by interpolation, cropping 128x128
    # for each scale:
    #   train_optimize to train network
    #   produce_test_image
    #   update training set

    t0 = time()

    source_image = load_image(image_path)
    father_scale_list = np.linspace(1,MAX_DOWNSCALE_FATHERS, DOWNSCALE_FATHERS_NUM)[1:]

    print("father_scale_list: {}".format(father_scale_list))

    interscale_list = np.linspace(1, desired_scale, num=INTER_SCALES_NUM+1)[1:] # not including 1

    print("interscale_list: {}".format(interscale_list))

    timestamp = get_logdir()
    print("Timestamp: {}".format(timestamp))
    logs_path = "./logs/" + timestamp + "/"

    # LR = tf.placeholder(tf.float32, shape=[None, CROP_SIZE, CROP_SIZE, INPUT_CHANNELS])
    HR_truth = tf.placeholder(tf.float32, shape=[None, CROP_SIZE, CROP_SIZE, INPUT_CHANNELS])

    LR = tf.placeholder(tf.float32, shape=[None, None, None, INPUT_CHANNELS])


    # test_image_placeholder = tf.placeholder(tf.float32, shape=[1,None, None, 1])

    with tf.name_scope('train'):
        HR_predictions = forward_prop(LR)
        with tf.name_scope('L1_loss'):
            loss = L1_loss(HR_predictions, HR_truth)
            tf.summary.scalar('L1_loss', loss)
        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


    #     with tf.name_scope('test'):
    #         current_image_placeholder = forward_prop(test_image_placeholder)

    # TB:
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(logs_path, sess.graph)


        sess.run(tf.global_variables_initializer())

        current_image = source_image.copy()

        last_scale = 1.0
        HR_fathers_list = []
        for current_scale in interscale_list:
            # with tf.name_scope('interscale_{}'.format(current_scale)):
            HR_fathers_list = HR_fathers_list + create_HR_fathers(current_image, father_scale_list)


            if verbose:
                print("fathers shapes:")
                for hf in HR_fathers_list:
                    print(hf.shape)


            LR_sons_list = create_LR_sons(HR_fathers_list, 1./current_scale,verbose=True)

            if verbose:
                print("sons shapes:")
                for ls in LR_sons_list:
                    print(ls.shape)


            HR_fathers_cropped_list, LR_sons_cropped_list = rand_crop_fathers_sons(HR_fathers_list, LR_sons_list, CROP_SIZE)
            # TODO take care of smaller than 128x128


            HR_fathers_cropped_tensor = np.expand_dims(np.array(HR_fathers_cropped_list),axis=3)
            LR_sons_cropped_tensor = np.expand_dims(np.array(LR_sons_cropped_list), axis=3)


            for i in range(ITERATIONS):
                sess.run(train_step, feed_dict={LR: LR_sons_cropped_tensor, HR_truth: HR_fathers_cropped_tensor})

                # if i % 100 == 0:
                if i % 1 == 0:
                    loss_empirical = sess.run(loss, feed_dict={LR: LR_sons_cropped_tensor, HR_truth: HR_fathers_cropped_tensor})
                    # TB:
                    summary = sess.run(merged, feed_dict={LR: LR_sons_cropped_tensor, HR_truth: HR_fathers_cropped_tensor})
                    train_writer.add_summary(summary, i)
                    print("Iteration {}, time elapsed: {} seconds, train cost: {}".format(i, time()- t0 ,loss_empirical))

            I_upscaled = cv2.resize(current_image, None, fx=current_scale/last_scale, fy=current_scale/last_scale, interpolation=UP_SAMPLING_METHOD)
            I_upscaled = np.expand_dims(I_upscaled, axis=0)
            I_upscaled = np.expand_dims(I_upscaled, axis=3)

            last_scale = current_scale # to calculate correct scaling of current_image

            current_image = sess.run(HR_predictions, feed_dict={LR: I_upscaled})

            # #deal with negative values:
            # image_min = np.min(current_image)
            # image_max = np.max(current_image)
            # current_image = (current_image - image_min)/(image_max-image_min)

            current_image = current_image[0, :, :, 0]


            # # maybe can replace by simple forward prop and later add merging of 8 images:
            # current_image = produce_test_image(current_image, current_scale).eval(session = sess)

            cv2.imwrite('./Results/abe{}.png'.format(current_scale), (current_image*255).astype('uint8'))

            # cv2.imshow('result scale {}'.format(current_scale), current_image)
            # cv2.waitKey(0)


    return current_image




def main(_):
    t0 = time()

    # result = ZSSR(EARTH_PATH, 2, verbose=True)
    result = ZSSR(ABE_PATH, 2, verbose=True)



    # cv2.imshow('result',result)


    print("Took {} seconds".format(time()-t0))

if __name__ == '__main__':
    tf.app.run(main=main)