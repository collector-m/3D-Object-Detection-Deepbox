import tensorflow as tf
import cv2, os
import numpy as np
from random import shuffle
import copy
from matplotlib import pyplot as plt

#####
#Training setting
BIN, OVERLAP = 2, 0.1
NORM_H, NORM_W = 224, 224
VEHICLES = ['Car', 'Truck', 'Van', 'Tram','Pedestrian','Cyclist']


def compute_anchors(angle):
    anchors = []
    
    wedge = 2.*np.pi/BIN
    l_index = int(angle/wedge)
    r_index = l_index + 1
    
    if (angle - l_index*wedge) < wedge/2 * (1+OVERLAP/2):
        anchors.append([l_index, angle - l_index*wedge])
        
    if (r_index*wedge - angle) < wedge/2 * (1+OVERLAP/2):
        anchors.append([r_index%BIN, angle - r_index*wedge])
        
    return anchors

def parse_annotation(label_dir, image_dir):
    all_objs = []
    dims_avg = {key:np.array([0, 0, 0]) for key in VEHICLES}
    dims_cnt = {key:0 for key in VEHICLES}
        
    for label_file in sorted(os.listdir(label_dir)):
        image_file = label_file.replace('txt', 'png')

        for line in open(label_dir + label_file).readlines():
            line = line.strip().split(' ')
            truncated = np.abs(float(line[1]))
            occluded  = np.abs(float(line[2]))

            if line[0] in VEHICLES and truncated < 0.1 and occluded < 0.1:
                new_alpha = float(line[3]) + np.pi/2.
                if new_alpha < 0:
                    new_alpha = new_alpha + 2.*np.pi
                new_alpha = new_alpha - int(new_alpha/(2.*np.pi))*(2.*np.pi)

                obj = {'name':line[0],
                       'image':image_file,
                       'xmin':int(float(line[4])),
                       'ymin':int(float(line[5])),
                       'xmax':int(float(line[6])),
                       'ymax':int(float(line[7])),
                       'dims':np.array([float(number) for number in line[8:11]]),
                       'new_alpha': new_alpha
                      }
                
                dims_avg[obj['name']]  = dims_cnt[obj['name']]*dims_avg[obj['name']] + obj['dims']
                dims_cnt[obj['name']] += 1
                dims_avg[obj['name']] /= dims_cnt[obj['name']]

                all_objs.append(obj)
    ###### flip data
    for obj in all_objs:
        # Fix dimensions
        obj['dims'] = obj['dims'] - dims_avg[obj['name']]

        # Fix orientation and confidence for no flip
        orientation = np.zeros((BIN,2))
        confidence = np.zeros(BIN)

        anchors = compute_anchors(obj['new_alpha'])

        for anchor in anchors:
            orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
            confidence[anchor[0]] = 1.

        confidence = confidence / np.sum(confidence)

        obj['orient'] = orientation
        obj['conf'] = confidence

        # Fix orientation and confidence for flip
        orientation = np.zeros((BIN,2))
        confidence = np.zeros(BIN)

        anchors = compute_anchors(2.*np.pi - obj['new_alpha'])
        for anchor in anchors:
            orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
            confidence[anchor[0]] = 1
            
        confidence = confidence / np.sum(confidence)

        obj['orient_flipped'] = orientation
        obj['conf_flipped'] = confidence
            
    return all_objs


def prepare_input_and_output(image_dir, train_inst, image_dir_right, train_inst_right):
    ### Prepare image patch
    xmin = train_inst['xmin'] #+ np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymin = train_inst['ymin'] #+ np.random.randint(-MAX_JIT, MAX_JIT+1)
    xmax = train_inst['xmax'] #+ np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymax = train_inst['ymax'] #+ np.random.randint(-MAX_JIT, MAX_JIT+1)

    xmin_right = train_inst_right['xmin'] #+ np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymin_right = train_inst_right['ymin'] #+ np.random.randint(-MAX_JIT, MAX_JIT+1)
    xmax_right = train_inst_right['xmax'] #+ np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymax_right = train_inst_right['ymax'] #+ np.random.randint(-MAX_JIT, MAX_JIT+1)

    if xmin_right < 0:
        xmin_right = 0;
    if ymin_right < 0:
        ymin_right = 0;

    xmin = xmin_right
    #xmax_right = xmax

    img = cv2.imread(image_dir + train_inst['image'])
    #print image_dir + train_inst['image']
    #print image_dir_right + train_inst_right['image']
    #print np.size(img, 0), np.size(img, 1), np.size(img, 2)
    #print "Here"
    #raw_input("Press Enter to continue...")
    #cv2.imshow('image', img)
    #raw_input("Press Enter to continue...")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = copy.deepcopy(img[ymin:ymax+1,xmin:xmax+1]).astype(np.float32)
    #print np.size(img, 0), np.size(img, 1), np.size(img, 2)
    #print "Here"
    #print xmin, ymin, xmax, ymax
    #print xmin_right, ymin_right, xmax_right, ymax_right
    
    img_right = cv2.imread(image_dir_right + train_inst_right['image'])
    #cv2.imshow('image_right', img_right)
    #raw_input("Press Enter to continue...")
    #cv2.imshow('image', img)
    #raw_input("Press Enter to continue...")
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
    img_right = copy.deepcopy(img_right[ymin_right:ymax_right+1,xmin_right:xmax_right+1]).astype(np.float32)


    
    #print img;
    #print train_inst;
    #print train_inst_right;
    # re-color the image
    #img += np.random.randint(-2, 3, img.shape).astype('float32')
    #t  = [np.random.uniform()]
    #t += [np.random.uniform()]
    #t += [np.random.uniform()]
    #t = np.array(t)

    #img = img * (1 + t)
    #img = img / (255. * 2.)

    # flip the image
    flip = np.random.binomial(1, .5)
    if flip > 0.5: 
        img = cv2.flip(img, 1)
        img_right = cv2.flip(img_right, 1)
        
    # resize the image to standard size
    img = cv2.resize(img, (NORM_H, NORM_W))
    #print np.amax(img)
    #print np.amin(img)
    #print type(img)
    cv2.imshow('image1', img/255)
    img = img - np.array([[[103.939, 116.779, 123.68]]])
    #print np.amax(img)
    #print np.amin(img)
    
    img_right = cv2.resize(img_right, (NORM_H, NORM_W))
    cv2.imshow('image1_right', img_right/255)
    img_right = img_right - np.array([[[103.939, 116.779, 123.68]]])
    #img = img[:,:,::-1]
    cv2.waitKey(0)
    raw_input("Press Enter to continue...")
    #cv2.destroyAllWindows()
    ### Fix orientation and confidence
    if flip > 0.5:
        return img, train_inst['dims'], train_inst['orient_flipped'], train_inst['conf_flipped'], img_right
    else:
        return img, train_inst['dims'], train_inst['orient'], train_inst['conf'], img_right

def data_gen(image_dir, all_objs, batch_size, image_dir_right, all_objs_right):
    num_obj = len(all_objs)
    
    keys = range(num_obj)
    np.random.shuffle(keys)
    
    l_bound = 0
    r_bound = batch_size if batch_size < num_obj else num_obj
    
    while True:
        if l_bound == r_bound:
            l_bound  = 0
            r_bound = batch_size if batch_size < num_obj else num_obj
            np.random.shuffle(keys)
        
        currt_inst = 0
        x_batch = np.zeros((r_bound - l_bound, 224, 224, 3))
        d_batch = np.zeros((r_bound - l_bound, 3))
        o_batch = np.zeros((r_bound - l_bound, BIN, 2))
        c_batch = np.zeros((r_bound - l_bound, BIN))

        x_batch_right = np.zeros((r_bound - l_bound, 224, 224, 3))
        
        for key in keys[l_bound:r_bound]:
            # augment input image and fix object's orientation and confidence

            image, dimension, orientation, confidence, image_right = prepare_input_and_output(image_dir, all_objs[key], image_dir_right, all_objs_right[key])
            #image_right, dimension_right, orientation_right, confidence_right = prepare_input_and_output(image_dir_right, all_objs_right[key])
            #plt.figure(figsize=(5,5))
            #plt.imshow(image/255./2.); plt.show()
            #print dimension
            #print orientation
            #print confidence
            
            x_batch[currt_inst, :] = image
            d_batch[currt_inst, :] = dimension
            o_batch[currt_inst, :] = orientation
            c_batch[currt_inst, :] = confidence
            
            x_batch_right[currt_inst, :] = image_right

            #plt.imshow(x_batch[currt_inst, :], interpolation='nearest')
            #plt.show()
            #plt.imshow(x_batch_right[currt_inst, :], interpolation='nearest')
            #plt.show()
            #cv2.imshow("DW", x_batch[currt_inst, :])
            #cv2.imshow("DW_right", x_batch_right[currt_inst, :]) 
            #print x_batch[currt_inst, :]
            #print x_batch[currt_inst, :].shape
            #raw_input("Press Enter to continue...")

            currt_inst += 1
                
        yield x_batch, [d_batch, o_batch, c_batch], x_batch_right
        
        l_bound  = r_bound
        r_bound = r_bound + batch_size
        if r_bound > num_obj: r_bound = num_obj

