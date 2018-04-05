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
cam = 3; % 2 = left color camera
%image_dir = fullfile(root_dir,[data_set '/image_' num2str(cam)], '_right');
%label_dir = fullfile(root_dir,[data_set '/label_' num2str(cam)]);
%label_dir = fullfile(root_dir, 'out');
image_dir = fullfile(root_dir, data_set , 'image_2_right');
label_dir = fullfile(root_dir, data_set, 'label_2');
calib_dir = fullfile(root_dir,[data_set '/calib']);

% get number of images for this dataset
nimages = length(dir(fullfile(image_dir, '*.png')));

% set up figure
h = visualization('init',image_dir);

% main loop
img_idx=0;

while 1

  % load projection matrix
  P = readCalibration(calib_dir,img_idx,cam);
  
  % load labels
  [objects, train_id] = readLabels(label_dir,img_idx);
  
  % visualization update for next frame
  visualization('update',image_dir,h,img_idx,nimages,data_set);
 
  right_label_file = fullfile('..','validation','label_2_right', train_id);
  right_file = fopen(right_label_file, 'wt');
  % for all annotated objects do
  for obj_idx=1:numel(objects)
   
    
    
    % plot 3D bounding box
    [corners,face_idx] = computeBox3D(objects(obj_idx),P);
    orientation = computeOrientation3D(objects(obj_idx),P);
    drawBox3D(h, objects(obj_idx),corners,face_idx,orientation);
    if size(corners,1) > 0
        min_coord = min(corners,[], 2);
        max_coord = max(corners,[], 2);
        % plot 2D bounding box
        %objects(obj_idx)
        objects(obj_idx).x1 = min_coord(1);
        objects(obj_idx).y1 = min_coord(2);
        objects(obj_idx).x2 = max_coord(1);
        objects(obj_idx).y2 = max_coord(2);
    end    
    
    drawBox2D(h,objects(obj_idx));
    
    fprintf(right_file, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n', ...
    objects(obj_idx).type, objects(obj_idx).truncation, objects(obj_idx).occlusion, objects(obj_idx).alpha, ...
    objects(obj_idx).x1, objects(obj_idx).y1, objects(obj_idx).x2, objects(obj_idx).y2, ...
    objects(obj_idx).h, objects(obj_idx).w, objects(obj_idx).l, objects(obj_idx).t, ...
    objects(obj_idx).ry);
    %corners
    
  end
  
  fclose(right_file);
  % force drawing and tiny user interface
  %{
  waitforbuttonpress; 
  key = get(gcf,'CurrentCharacter');
  switch lower(key)                         
    case 'q',  break;                                 % quit
    case '-',  img_idx = max(img_idx-1,  0);          % previous frame
    case 'x',  img_idx = min(img_idx+1000,nimages-1); % +100 frames
    case 'y',  img_idx = max(img_idx-1000,0);         % -100 frames
    otherwise, img_idx = min(img_idx+1,  nimages-1);  % next frame
  end 
  %}
  img_idx = min(img_idx+1,  nimages-1);  % next frame
end

% clean up
close all;
