%% test_images2video
imagefile_dir = strcat(datadir,'_',string(interpAlgorithm),'_morph_');
videofile_name = strcat(datadir,'video_morph_',string(interpAlgorithm),'_frames_',string(numframes));

if exist('videofile_name','file')
    delete videofile_name.avi
end

aviobj = VideoWriter(videofile_name,'Motion JPEG AVI');
aviobj.FrameRate = video_fps;

open(aviobj);

for i = 0:numframes-1
    imagefile_name = strcat(imagefile_dir,sprintf('%d',i),'.png');
    frames1 = imread(imagefile_name);
    writeVideo(aviobj,frames1);
    fprintf('%d completed\n',i);
end

close(aviobj);

fprintf('video completed\n');