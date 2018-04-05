%% Compute coordinates of box in right image
clear all; close all;

%% Extract the index of training and validation images
fileID = fopen('../ImageSets/train.txt','r');
i = 1;
tline = fgetl(fileID);
while ischar(tline)
    train_id{i} = tline;
    tline = fgetl(fileID);
    i = i + 1;
end

left_train_img_dir = fullfile('..','training', 'image_2');

for i = 1:size(train_id, 2):
    left_img_file = fullfile('training','left-images-with-disparity', strcat(train_id{i},'.png'));
    left_img = imread(left_img_file); 
end