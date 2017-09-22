clear
clc

% Compute all GIST
for scene = [8 15];
    count=1;
    fileName = sprintf('datasets/%d_scene/filenames_scene%d.txt',scene,scene);
    fileID = fopen(fileName,'wt');
    for label = 1:scene;
        file_path =  sprintf('datasets/%d_scene/images/Label%d',scene,label);
        img_path_list = dir(file_path);
                
        for j = 3: length(img_path_list)
            image_name = img_path_list(j).name;
            fprintf(fileID,'img%d_%s\n',count,image_name);
            img = imread(sprintf('%s/%s',file_path,image_name));
            count = count+1;
        end
    end
end