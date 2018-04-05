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
     left_img_file = fullfile('..','training','image_2', strcat(train_id{i},'.png'));
     left_img = imread(left_img_file);
     right_img_file = fullfile('..','training','image_2_right', strcat(train_id{i},'.png'));
     right_img = imread(right_img_file);
    % figure (1)
    % imtool(stereoAnaglyph(left_img,right_img));
     left_img = rgb2gray(left_img);
     right_img = rgb2gray(right_img);
     
     disparityMap = disparity(left_img, right_img, 'DisparityRange',[0 64], 'BlockSize', 55); 
     figure (2)
     imshow(disparityMap);
     
     %% Pull up block coordinates of left image
     left_label_file = fullfile('..','training','label_2', strcat(train_id{i},'.txt'));
     left_file = fopen(left_label_file);
     left_block_coord = textscan(left_file, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
     right_block_coord = left_block_coord;
     
     right_label_file = fullfile('..','training','label_2_right', strcat(train_id{i},'.txt'));
     right_file = fopen(right_label_file, 'wt');
     %for j = 1:size(left_block_coord{1}, 1)
     for j = 1:1
        x_min = left_block_coord{5}(j);
        y_min = left_block_coord{6}(j); 
        x_max = left_block_coord{7}(j);  
        y_max = left_block_coord{8}(j);
        disparityMap(int8(y_min), int8(x_min)), 
        disparityMap(int8(y_max), int8(x_max))
        right_block_coord{5}(j) = x_min - disparityMap(int8(y_min), int8(x_min));
        right_block_coord{6}(j) = left_block_coord{6}(j);
        right_block_coord{7}(j) = x_max - disparityMap(int8(y_max), int8(x_max));
        right_block_coord{8}(j) = left_block_coord{8}(j);
        fprintf(right_file, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n', ...
        right_block_coord{1}{j}, right_block_coord{2}(j), ...
        right_block_coord{3}(j), right_block_coord{4}(j), right_block_coord{5}(j), ...
        right_block_coord{6}(j), right_block_coord{7}(j), right_block_coord{8}(j), ...
        right_block_coord{9}(j), right_block_coord{10}(j), right_block_coord{11}(j), ...
        right_block_coord{12}(j), right_block_coord{13}(j), right_block_coord{14}(j), ...
        right_block_coord{15}(j));
        
        subplot(1,2,1);
        imshow(left_img);
        hold on;
        rectangle('Position', [x_min y_min x_max - x_min y_max - y_min], 'EdgeColor', 'r', 'LineWidth', 3);
        
        
        subplot(1,2,2);
        imshow(right_img);
        hold on;
        rectangle('Position', [right_block_coord{5}(j) right_block_coord{6}(j) ...
            right_block_coord{7}(j) - right_block_coord{5}(j) right_block_coord{8}(j) - right_block_coord{6}(j)], ...
            'EdgeColor', 'r', 'LineWidth', 3);
        %hold off;
        ccc = 1;
        
     end
     fclose(right_file);
     
     
     
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

