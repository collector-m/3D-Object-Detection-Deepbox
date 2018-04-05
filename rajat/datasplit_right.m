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

train_img_dir = fullfile('training', 'image_2_right');
val_img_dir = fullfile('validation', 'image_2_right');
for i = 1:size(train_id, 2)
     img_dir = fullfile('..','training','image_2_right', strcat(train_id{i},'.png'));
     copyfile(img_dir, train_img_dir);
end


for i = 1:size(val_id, 2)
    img_dir = fullfile('..','training','image_2_right', strcat(val_id{i},'.png'));
    copyfile(img_dir, val_img_dir);
end