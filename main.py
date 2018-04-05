import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2, os
import numpy as np
import time
from random import shuffle
from data_processing import *
import sys
import argparse
from tqdm import tqdm

import random
#####
#Training setting

BIN, OVERLAP = 2, 0.1
W = 1.
ALPHA = 1.
MAX_JIT = 3
NORM_H, NORM_W = 224, 224
VEHICLES = ['Car', 'Truck', 'Van', 'Tram','Pedestrian','Cyclist']
BATCH_SIZE = 8
learning_rate = 0.0001
epochs = 50
save_path = './model/'

dims_avg = {'Cyclist': np.array([ 1.73532436,  0.58028152,  1.77413709]), 'Van': np.array([ 2.18928571,  1.90979592,  5.07087755]), 'Tram': np.array([  3.56092896,   2.39601093,  18.34125683]), 'Car': np.array([ 1.52159147,  1.64443089,  3.85813679]), 'Pedestrian': np.array([ 1.75554637,  0.66860882,  0.87623049]), 'Truck': np.array([  3.07392252,   2.63079903,  11.2190799 ])}


#### Placeholder
inputs = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
inputs_right = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
d_label = tf.placeholder(tf.float32, shape = [None, 3])
o_label = tf.placeholder(tf.float32, shape = [None, BIN, 2])
c_label = tf.placeholder(tf.float32, shape = [None, BIN])


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='3D bounding box')
    parser.add_argument('--mode',dest = 'mode',help='train or test',default = 'test')
    parser.add_argument('--image',dest = 'image',help='Image path')
    parser.add_argument('--label',dest = 'label',help='Label path')
    parser.add_argument('--box2d',dest = 'box2d',help='2D detection path')
    parser.add_argument('--output',dest = 'output',help='Output path', default = './validation/result_2/')
    parser.add_argument('--model',dest = 'model')
    parser.add_argument('--gpu',dest = 'gpu',default= '0')
    parser.add_argument('--label_right',dest = 'label_right',help='Image path')
    parser.add_argument('--image_right',dest = 'image_right',help='Image path')
    parser.add_argument('--box2d_right',dest = 'box2d_right',help='2D detection path')
    args = parser.parse_args()

    return args


def build_model():

  #### build some layer 
  def LeakyReLU(x, alpha):
      return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

  def orientation_loss(y_true, y_pred):
      # Find number of anchors
      anchors = tf.reduce_sum(tf.square(y_true), axis=2)
      anchors = tf.greater(anchors, tf.constant(0.5))
      anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)

      # Define the loss
      loss = (y_true[:,:,0]*y_pred[:,:,0] + y_true[:,:,1]*y_pred[:,:,1])
      loss = tf.reduce_sum((2 - 2 * tf.reduce_mean(loss,axis=0))) / anchors

      return tf.reduce_mean(loss)

  #####
  #Build Graph
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):

    net_left = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    net_left = slim.max_pool2d(net_left, [2, 2], scope='pool1')
    net_left = slim.repeat(net_left, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net_left = slim.max_pool2d(net_left, [2, 2], scope='pool2')
    net_left = slim.repeat(net_left, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net_left = slim.max_pool2d(net_left, [2, 2], scope='pool3')
    net_left = slim.repeat(net_left, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    net_left = slim.max_pool2d(net_left, [2, 2], scope='pool4')
    net_left = slim.repeat(net_left, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    net_left = slim.max_pool2d(net_left, [2, 2], scope='pool5')


    net_right = slim.repeat(inputs_right, 2, slim.conv2d, 64, [3, 3], scope='conv6')
    net_right = slim.max_pool2d(net_right, [2, 2], scope='pool6')
    net_right = slim.repeat(net_right, 2, slim.conv2d, 128, [3, 3], scope='conv7')
    net_right = slim.max_pool2d(net_right, [2, 2], scope='pool7')
    net_right = slim.repeat(net_right, 3, slim.conv2d, 256, [3, 3], scope='conv8')
    net_right = slim.max_pool2d(net_right, [2, 2], scope='pool8')
    net_right = slim.repeat(net_right, 3, slim.conv2d, 512, [3, 3], scope='conv9')
    net_right = slim.max_pool2d(net_right, [2, 2], scope='pool9')
    net_right = slim.repeat(net_right, 3, slim.conv2d, 512, [3, 3], scope='conv10')
    net_right = slim.max_pool2d(net_right, [2, 2], scope='pool10')

    #conv5_left = tf.contrib.layers.flatten(net_left)

    #conv5_right = tf.contrib.layers.flatten(net_right)

    #dimension = slim.fully_connected(conv5, 512, scope='fc7_d')
    net = tf.concat([net_left, net_right], 3)
    conv5 = tf.contrib.layers.flatten(net)


    dimension = slim.fully_connected(conv5, 512, activation_fn=None, scope='fc7_d')
    dimension = LeakyReLU(dimension, 0.1)
    dimension = slim.dropout(dimension, 0.5, scope='dropout7_d')
    #dimension = slim.fully_connected(dimension, 3, scope='fc8_d')
    dimension = slim.fully_connected(dimension, 3, activation_fn=None, scope='fc8_d')
    #dimension = LeakyReLU(dimension, 0.1)

    #loss_d = tf.reduce_mean(tf.square(d_label - dimension))
    loss_d = tf.losses.mean_squared_error(d_label, dimension)

    #orientation = slim.fully_connected(conv5, 256, scope='fc7_o')
    orientation = slim.fully_connected(conv5, 256, activation_fn=None, scope='fc7_o')
    orientation = LeakyReLU(orientation, 0.1)
    orientation = slim.dropout(orientation, 0.5, scope='dropout7_o')
    #orientation = slim.fully_connected(orientation, BIN*2, scope='fc8_o')
    orientation = slim.fully_connected(orientation, BIN*2, activation_fn=None, scope='fc8_o')
    #orientation = LeakyReLU(orientation, 0.1)
    orientation = tf.reshape(orientation, [-1, BIN, 2])
    orientation = tf.nn.l2_normalize(orientation, dim=2)
    loss_o = orientation_loss(o_label, orientation)

    #confidence = slim.fully_connected(conv5, 256, scope='fc7_c')
    confidence = slim.fully_connected(conv5, 256, activation_fn=None, scope='fc7_c')
    confidence = LeakyReLU(confidence, 0.1)
    confidence = slim.dropout(confidence, 0.5, scope='dropout7_c')
    confidence = slim.fully_connected(confidence, BIN, activation_fn=None, scope='fc8_c')
    loss_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=c_label, logits= confidence))
   
    confidence = tf.nn.softmax(confidence)
    #loss_c = tf.reduce_mean(tf.square(c_label - confidence))
    #loss_c = tf.losses.mean_squared_error(c_label, confidence)
    
    total_loss = 4. * loss_d + 8. * loss_o + loss_c
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
    
    return dimension, orientation, confidence, total_loss, optimizer, loss_d, loss_o, loss_c


def train(image_dir, box2d_loc, label_dir, label_dir_right, image_dir_right):

    # load data & gen data
    all_objs = parse_annotation(label_dir, image_dir)
    all_objs_right = parse_annotation(label_dir_right, image_dir_right)
    all_exams  = len(all_objs)
    #print all_objs
    #raw_input("Press Enter to continue...")
    #print all_objs_right
    #raw_input("Press Enter to continue...")
    #print all_exams
    #raw_input("Press Enter to continue...")
    #print all_objs
    #print type(all_objs)
    #randomize = range(all_exams) 
    #print randomize
    #random.shuffle(randomize)
    #print randomize
    #print all_objs[1]
    #all_objs = all_objs[randomize]
    #all_objs_right = all_objs_right[randomize]
    #np.random.shuffle(all_objs)
    two_lists_combined = list(zip(all_objs, all_objs_right))
    random.shuffle(two_lists_combined)
    all_objs, all_objs_right = zip(*two_lists_combined)
    #raw_input("Press Enter to continue...")
    #raw_input("Press Enter to continue...")
    train_gen = data_gen(image_dir, all_objs, BATCH_SIZE, image_dir_right, all_objs_right)
    #train_gen_right = data_gen(image_dir_right, all_objs_right, BATCH_SIZE)
    train_num = int(np.ceil(all_exams/BATCH_SIZE))
    
    ### buile graph
    dimension, orientation, confidence, loss, optimizer, loss_d, loss_o, loss_c = build_model()

    ### GPU config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    # create a folder for saving model
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    variables_to_restore = slim.get_variables()[:52] ## vgg16-conv5
#    print variables_to_restore
#    print "Here"
    """
    def name_in_checkpoint(var):
        if "conv1" in var.op.name:
            return var.op.name.replace("conv1", "conv6")
        if "conv2" in var.op.name:
            return var.op.name.replace("conv2", "conv7")
        if "conv3" in var.op.name:
            return var.op.name.replace("conv3", "conv8")
        if "conv4" in var.op.name:
            return var.op.name.replace("conv4", "conv9")
        if "conv5" in var.op.name:
            return var.op.name.replace("conv5", "conv10")
    """
    #variables_to_restore_right = slim.get_variables()[27:52] ## vgg16-conv5
#    variables_to_restore_right = {name_in_checkpoint(var):var for var in variables_to_restore}
    saver = tf.train.Saver(max_to_keep=100)

#    print variables_to_restore_right
#    print "Here2"
    #Load pretrain VGG model
    ckpt_list = tf.contrib.framework.list_variables('./vgg_16.ckpt')[1:-7]
#    print(ckpt_list)
#    print(len(ckpt_list))
#    print variables_to_restore[0]
#    print variables_to_restore[1]
#    print variables_to_restore[2]
#    print variables_to_restore[3]
#    print variables_to_restore[4]
#    print variables_to_restore[5]
#    print variables_to_restore[6]
#    print variables_to_restore[7]
#    print variables_to_restore[9]
#    print variables_to_restore[10]
    for name in range(1,len(ckpt_list),2):
        tf.contrib.framework.init_from_checkpoint('./vgg_16.ckpt', {ckpt_list[name-1][0]: variables_to_restore[name]})
        tf.contrib.framework.init_from_checkpoint('./vgg_16.ckpt', {ckpt_list[name][0]: variables_to_restore[name-1]})
#        print name - 1, ckpt_list[name-1][0], variables_to_restore[name]
#        print name, ckpt_list[name][0], variables_to_restore[name-1]
        tf.contrib.framework.init_from_checkpoint('./vgg_16.ckpt', {ckpt_list[name-1][0]: variables_to_restore[name + 26]})
        tf.contrib.framework.init_from_checkpoint('./vgg_16.ckpt', {ckpt_list[name][0]: variables_to_restore[name + 26 - 1]})
        
#        print name - 1, ckpt_list[name-1][0], variables_to_restore[name + 26]
#        print name, ckpt_list[name][0], variables_to_restore[name + 26 - 1]
#        print name -1, ckpt_list[name-1]
#        print name, ckpt_list[name]

    #raise IOError(('Image not found.'))
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)


    # Start to train model
    for epoch in range(epochs):
        epoch_loss = np.zeros((train_num,1),dtype = float)
        tStart_epoch = time.time()
        batch_loss = 0.0
        for num_iters in tqdm(range(train_num),ascii=True,desc='Epoch '+str(epoch+1)+' : Loss:'+str(batch_loss)):
            train_img, train_label, train_img_right = train_gen.next()
            #train_img_right, train_label_right = train_gen_right.next()
            _,batch_loss = sess.run([optimizer,loss], feed_dict={inputs: train_img, inputs_right: train_img_right, d_label: train_label[0], o_label: train_label[1], c_label: train_label[2]})

            epoch_loss[num_iters] = batch_loss 

        # save model
        if (epoch+1) % 5 == 0:
            saver.save(sess,save_path+"model", global_step = epoch+1)

        # Print some information
        print "Epoch:", epoch+1, " done. Loss:", np.mean(epoch_loss)
        tStop_epoch = time.time()
        print "Epoch Time Cost:", round(tStop_epoch - tStart_epoch,2), "s"
        sys.stdout.flush()

def test(model, image_dir, box2d_loc, box3d_loc, image_dir_right, box2d_loc_right):

    ### buile graph
    dimension, orientation, confidence, loss, optimizer, loss_d, loss_o, loss_c = build_model()

    ### GPU config 
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Restore model
    saver = tf.train.Saver()
    saver.restore(sess, model)

    # create a folder for saving result
    if os.path.isdir(box3d_loc) == False:
        os.mkdir(box3d_loc)

    # Load image & run testing
    all_image = sorted(os.listdir(image_dir))

    for f in all_image:
        image_file = image_dir + f
        image_file_right = image_dir_right + f
        box2d_file = box2d_loc + f.replace('png', 'txt')
        box3d_file = box3d_loc + f.replace('png', 'txt')

        box2d_file_right = box2d_loc_right + f.replace('png', 'txt')
        print image_file
        with open(box3d_file, 'w') as box3d:
            img = cv2.imread(image_file)
            img = img.astype(np.float32, copy=False)
            img_right = cv2.imread(image_file_right)
            img_right = img_right.astype(np.float32, copy=False)
            file_right = open(box2d_file_right)
            for line in open(box2d_file):

                
                line_right = file_right.readline()
                #print(line)
                #print(line_right)
                #raw_input("Press Enter to continue...")

                line = line.strip().split(' ')
                line_right = line_right.strip().split(' ')
                #print(line)
                #print(line_right)
                #raw_input("Press Enter to continue...")
                truncated = np.abs(float(line[1]))
                occluded  = np.abs(float(line[2]))

                obj = {'xmin':int(float(line[4])),
                       'ymin':int(float(line[5])),
                       'xmax':int(float(line[6])),
                       'ymax':int(float(line[7])),
                       }

                obj_right = {'xmin':int(float(line_right[4])),
                             'ymin':int(float(line_right[5])),
                             'xmax':int(float(line_right[6])),
                             'ymax':int(float(line_right[7])),
                            }

                if (obj_right['xmin'] < 0):
                    obj_right['xmin'] = 0
                if (obj_right['ymin'] < 0):
                    obj_right['ymin'] = 0

                patch = img[obj['ymin']:obj['ymax'],obj['xmin']:obj['xmax']]
                patch = cv2.resize(patch, (NORM_H, NORM_W))
                patch = patch - np.array([[[103.939, 116.779, 123.68]]])
                patch = np.expand_dims(patch, 0)

                patch_right = img_right[obj_right['ymin']:obj_right['ymax'],obj_right['xmin']:obj_right['xmax']]
                patch_right = cv2.resize(patch_right, (NORM_H, NORM_W))
                patch_right = patch_right - np.array([[[103.939, 116.779, 123.68]]])
                patch_right = np.expand_dims(patch_right, 0)
                prediction = sess.run([dimension, orientation, confidence], feed_dict={inputs: patch, inputs_right: patch_right})


                # Transform regressed angle
                max_anc = np.argmax(prediction[2][0])
                anchors = prediction[1][0][max_anc]

                if anchors[1] > 0:
                    angle_offset = np.arccos(anchors[0])
                else:
                    angle_offset = -np.arccos(anchors[0])

                wedge = 2.*np.pi/BIN
                angle_offset = angle_offset + max_anc*wedge
                angle_offset = angle_offset % (2.*np.pi)

                angle_offset = angle_offset - np.pi/2
                if angle_offset > np.pi:
                    angle_offset = angle_offset - (2.*np.pi)

                line[3] = str(angle_offset)
                 
                line[-1] = angle_offset +np.arctan(float(line[11]) / float(line[13]))
                
                # Transform regressed dimension
                if line[0] in VEHICLES:
                    dims = dims_avg[line[0]] + prediction[0][0]
                else:
                    dims = dims_avg['Car'] + prediction[0][0]

                line = line[:8] + list(dims) + line[11:]
                
                # Write regressed 3D dim and oritent to file
                line = ' '.join([str(item) for item in line]) +' '+ str(np.max(prediction[2][0]))+ '\n'
                box3d.write(line)

 

if __name__ == "__main__":
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.image is None:
        raise IOError(('Image not found.'.format(args.image)))

    if args.box2d is None :
        raise IOError(('2D bounding box not found.'.format(args.box2d)))

    if args.image_right is None:
        raise IOError(('Image not found.'.format(args.image_right))) 

    #if args.box2d_right is None:
    #    raise IOError(('Image not found.'.format(args.image_right))) 

    if args.mode == 'train':
        if args.label is None:
            raise IOError(('Label not found.'.format(args.label)))

        train(args.image, args.box2d, args.label, args.label_right, args.image_right)
    else:
        if args.model is None:
            raise IOError(('Model not found.'.format(args.model)))
           

        test(args.model, args.image, args.box2d, args.output, args.image_right, args.box2d_right)

