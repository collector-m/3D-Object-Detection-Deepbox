% KITTI OBJECT DETECTION AND ORIENTATION ESTIMATION BENCHMARK DEMONSTRATION
% 
% This tool displays the images and the object labels for the benchmark and
% provides an entry point for writing your own interface to the data set.
% Before running this tool, set root_dir to the directory where you have
% downloaded the dataset. 'root_dir' must contain the subdirectory
% 'training', which in turn contains 'image_2', 'label_2' and 'calib'.
% For more information about the data format, please look into readme.txt.
%
% Usage:
%   SPACE: next frame
%   '-':   last frame
%   'x':   +10 frames
%   'y':   -10 frames
%   q:     quit
%
% Occlusion Coding:
%   green:  not occluded
%   yellow: partly occluded
%   red:    fully occluded
%   white:  unknown
%
% Truncation Coding:
%   solid:  not truncated
%   dashed: truncated

% clear and close everything
clear all; close all;
disp('======= KITTI DevKit Demo =======');

% options
root_dir = '../';
data_set = 'validation';
 
% get sub-directories
cam = 2; % 2 = left color camera
image_dir = fullfile(root_dir,[data_set '/image_' num2str(cam)]);
label_dir_1 = fullfile(root_dir,[data_set '/label_' num2str(cam)]);
label_dir_2 = fullfile(root_dir, 'models-to-test','original-output','out');
label_dir_3 = fullfile(root_dir, 'models-to-test','output-with-stereo-images','out');
label_dir_4 = fullfile(root_dir, 'models-to-test','output-with-stereo-images-with-disparity','out');
calib_dir = fullfile(root_dir,[data_set '/calib']);

% get number of images for this dataset
nimages = length(dir(fullfile(image_dir, '*.png')));

% set up figure
h = visualization_all_images('init',image_dir);

% main loop
img_idx=0;
count = 1;
while 1

  % load projection matrix
  P = readCalibration(calib_dir,img_idx,cam);
  
  % load labels
  [objects_1, train_id] = readLabels(label_dir_1,img_idx);
  [objects_2, train_id] = readLabels(label_dir_2,img_idx);
  [objects_3, train_id] = readLabels(label_dir_3,img_idx);
  [objects_4, train_id] = readLabels(label_dir_4,img_idx);
   
  
  % visualization update for next frame
  visualization_all_images('update',image_dir,h, img_idx,nimages,data_set, count);
 
  % for all annotated objects do
  for obj_idx=1:numel(objects_1)
   
    % plot 2D bounding box
    drawBox2D(h,objects_1(obj_idx));
    
    % plot 3D bounding box
    [corners_1,face_idx_1] = computeBox3D(objects_1(obj_idx),P);
    [corners_2,face_idx_2] = computeBox3D(objects_2(obj_idx),P);
    [corners_3,face_idx_3] = computeBox3D(objects_3(obj_idx),P);
    [corners_4,face_idx_4] = computeBox3D(objects_4(obj_idx),P);
    orientation_1 = computeOrientation3D(objects_1(obj_idx),P);
    orientation_2 = computeOrientation3D(objects_2(obj_idx),P);
    orientation_3 = computeOrientation3D(objects_3(obj_idx),P);
    orientation_4 = computeOrientation3D(objects_4(obj_idx),P);
    drawBox3D_for_all_outputs(h, objects_1(obj_idx),corners_1,face_idx_1,orientation_1, 2);
    drawBox3D_for_all_outputs(h, objects_2(obj_idx),corners_2,face_idx_2,orientation_2, 3);
    drawBox3D_for_all_outputs(h, objects_3(obj_idx),corners_3,face_idx_3,orientation_3, 4);
    drawBox3D_for_all_outputs(h, objects_4(obj_idx),corners_4,face_idx_4,orientation_4, 5);
    
  end
  count = count + 1;
  % force drawing and tiny user interface
  waitforbuttonpress; 
  key = get(gcf,'CurrentCharacter');
  switch lower(key)                         
    case 'q',  break;                                 % quit
    case '-',  img_idx = max(img_idx-1,  0);          % previous frame
    case 'x',  img_idx = min(img_idx+1000,nimages-1); % +100 frames
    case 'y',  img_idx = max(img_idx-1000,0);         % -100 frames
    otherwise, img_idx = min(img_idx+1,  nimages-1);  % next frame
  end
  
end

% clean up
close all;
