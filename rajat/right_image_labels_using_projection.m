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
i = 1;
fileID = fopen('../ImageSets/val.txt','r');
tline = fgetl(fileID);
while ischar(tline)
    val_id{i} = tline;
    tline = fgetl(fileID);
    i = i + 1;
end


left_train_img_dir = fullfile('..','training', 'image_2');
left_val_img_dir = fullfile('..','validation', 'image_2');

right_train_img_dir = fullfile('..','training', 'image_2_right');
right_val_img_dir = fullfile('..','validation', 'image_2_right');

for i = 1:size(train_id, 2)
    % Pull up images
    % left_img_file = fullfile('..','training','image_2', strcat(train_id{i},'.png'));
    % left_img = imread(left_img_file);
     right_img_file = fullfile('..','training','image_2_right', strcat(train_id{i},'.png'));
     right_img = imread(right_img_file);
    
     
     
     %% Pull up block coordinates of left image
     right_label_file = fullfile('..','training','label_2_right', strcat(train_id{i},'.txt'));
     right_file = fopen(right_label_file);
     right_block_coord = textscan(right_file, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
     %right_block_coord = left_block_coord;
     
     %right_label_file = fullfile('..','training','label_2_right', strcat(train_id{i},'.txt'));
     %right_file = fopen(right_label_file, 'wt');
     %for j = 1:size(left_block_coord{1}, 1)
     imshow(right_img);
     hold on;
     for j = 1:size(right_block_coord{1}, 1)
        x_min = right_block_coord{5}(j);
        y_min = right_block_coord{6}(j); 
        x_max = right_block_coord{7}(j);  
        y_max = right_block_coord{8}(j);
        
        
        
        rectangle('Position', [x_min y_min x_max - x_min y_max - y_min], 'EdgeColor', 'r', 'LineWidth', 3);
       
        
     end
     %fclose(right_file);
     
     
     
end






        %{
        j = 0;
        new_disparity = disparity;
        if (x_min - disparity - 7 < 0)
            new_disparity = x_min - 0 - 7;
        end
        sad = zeros(new_disparity, block_size * block_size)
        for j = 0:new_disparity
            sad[j] = 
        end
        right_x_min = min_arg(sad);
        
        sad = zeros(new_disparity, block_size * block_size);
        for j = 0:new_disparity
            sad[j] = 
        end
        right_x_max = min_arg(sad);
        %}

