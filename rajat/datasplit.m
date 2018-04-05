clear all; close all;
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
%img_dir = fullfile('..','training','image_2')
%label_dir = fullfile('..', 'training', 'label_2');
%calib_dir = fullfile('..','training', 'calib');
train_img_dir = fullfile('training', 'image_2');
train_label_dir = fullfile('training', 'label_2');
train_calib_dir = fullfile('training', 'calib');
val_img_dir = fullfile('validation', 'image_2');
val_label_dir = fullfile('validation', 'label_2');
val_calib_dir = fullfile('validation','calib');
%for i = 1:size(train_id, 2)
%    %file_no = num2str(train_id(i))
%    img_dir = fullfile('..','training','image_2', strcat(train_id{i},'.png'));
%    label_dir = fullfile('..', 'training', 'label_2', strcat(train_id{i}, '.txt'));
%    calib_dir = fullfile('..','training', 'calib', strcat(train_id{i}, '.txt'));
%    copyfile(img_dir, train_img_dir);
%    copyfile (label_dir, train_label_dir);
%    copyfile (calib_dir, train_calib_dir);
%end


for i = 1:size(val_id, 2)
    %file_no = num2str(train_id(i))
    img_dir = fullfile('..','training','image_2', strcat(val_id{i},'.png'));
    label_dir = fullfile('..', 'training', 'label_2', strcat(val_id{i}, '.txt'));
    calib_dir = fullfile('..','training', 'calib', strcat(val_id{i}, '.txt'));
    copyfile(img_dir, val_img_dir);
    copyfile (label_dir, val_label_dir);
    copyfile (calib_dir, val_calib_dir);
end