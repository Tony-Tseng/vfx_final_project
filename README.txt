# Install the enviornment by requirements.txt
conda install --file requirements.txt

# How to run
1. To run image cloning
cd code && python app_cloning.py

2. To run image cloning & video tracking
cd code && python app_tracking.py

# Folder Structure:
code: 程式碼

image: 影像
   |
   |-- source.jpg: The images that we want to clone.
   |   
   |-- target.jpg: The images that we want to stick at.
   |
   |-- result.png: The result images with image cloning.

Video: 影片
   |
   |-- ice.mp4/ sand.mp4: The video that we want to clone.
   |   
   |-- result_ice.mp4/ result_sand.mp4: The result video.

Reference code:
1. https://github.com/ZheyuanXie/KLT-Feature-Tracking
2. https://github.com/fafa1899/MVCImageBlend/blob/master/ImgViewer/qimageshowwidget.cpp
3. https://github.com/pablospe/MVCDemo/blob/master/src/MVCCloner.cpp
4. https://github.com/apwan/MVC

Reference Paper: 
1. https://www.cs.huji.ac.il/~danix/mvclone/
2. https://github.com/fafa1899/MVCImageBlend
3. https://github.com/pablospe/MVCDemo
4. https://github.com/apwan/MVC
