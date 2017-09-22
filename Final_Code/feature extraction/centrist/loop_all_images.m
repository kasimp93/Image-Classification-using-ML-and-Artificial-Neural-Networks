clear
clc

% GIST Parameters:
clear param
param.imageSize = [256 256]; % it works also with non-square images
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Compute all GIST
for scene = [8 15];
    count=1;
    if (scene==8)
        X_cent = zeros(2688,254);
        Y_cent = zeros(2688,1);
    else
        X_cent = zeros(4485,254);
        Y_cent = zeros(4485,1);
    end
    for label = 1:scene;
        file_path =  sprintf('datasets/%d_scene/images/Label%d',scene,label);
        img_path_list = dir(file_path);
        
        fileName = sprintf('datasets/%d_scene/images/filenames_label%d.txt',scene,label);
        fileID = fopen(fileName,'wt');
        
        for j = 3: length(img_path_list)
            image_name = img_path_list(j).name;
            fprintf(fileID,'img%d_%s\n',j-2,image_name);
            img = imread(sprintf('%s/%s',file_path,image_name));
            X_cent(count,:) = centrist(img);
            Y_cent(count) = label;
            count = count+1;
        end
        fprintf('CENTRIST computed for %d-scene %d-label\n',scene,label); 
    end
    save(sprintf('datasets/%d_scene/X_cent',scene),'X_cent');
    save(sprintf('datasets/%d_scene/Y_cent',scene),'Y_cent');
end