## 一、关键demo介绍
- yolov5_distince.py :基于yolov5社交距离
- yolov5_flask.py :基于yolov5+flask 网页端实现
- yolov5_flask_distince.py :基于yolov5+flask+社交距离 网页端实现
- camera.py :基于yolov5+deepsort（目标跟踪）+Flask Video Streaming实现浏览器打开摄像头 进行目标跟踪

## 二、安装与使用

#### 1、部署环境
pip install -r requirements.txt

#### 2、运行（无flask版本）
>python xxx.py --source xxx

xxx.py :指"一"中代码文件

xxx :指视频路径（自备视频文件）

#### 3、运行（含flask版本）
- 在根目录下的app.py 修改from yolov5_flask import Camera 的包名（根据自己需求，这里采用yolov5_flask.py）
    - 在浏览器使用摄像头就用：from camera import Camera
- 运行:flask run
- 浏览器访问:http://127.0.0.1:5000/index即可

如果你有好的想法，欢迎发issue，或者参与本开源项目。
