#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : linjie
from flask import Flask, render_template, Response,  request, session, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from detect import *
import os
import torch
from importlib import import_module
# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera
from flask_cors import *

app = Flask(__name__)
# UPLOAD_FOLDER = "C:\Users\Arpit Sharma\Desktop\Friendship goals\content\yolov5\static\uploads"
DETECTION_FOLDER = r'./static/detections'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
#app.config['DETECTION_FOLDER'] = DETECTION_FOLDER


# @app.route("/")
# def index():
#     return render_template("index1.html")
#
# @app.route("/about")
# def about():
#     print("I about")
#     return render_template("about.html")
#
# @app.route("/uploaded",methods = ["GET","POST"])
# def uploaded():
#     if request.method == 'POST':
#         f = request.files['file']
#         filename = secure_filename(f.filename)
#         print(filename)
#         file_path = os.path.join(r"./static/uploads", filename)
#         print(file_path)
#         f.save(file_path)
#         with torch.no_grad():
#             detect_image(file_path, DETECTION_FOLDER)
#         return render_template("uploaded.html",display_detection = filename, fname = filename)

# @app.route("/video")
# def about2():
#     print("I about")
#     os.system('python detect.py --source 0')
#     return render_template("about.html")

@app.route('/woer/detection')
def index():
    """Video streaming home page."""
    return render_template('woer/detection.html')

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        # a = camera.people_appeal()
        # print('a:{}0'.format(a))
        # for i in a:
        #     if i =='people':
        #         print('是people：{}}')
        #         people_appeal()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




'''前台页面跳转'''
@app.route('/woer/index')
def woer_index():
    return render_template('woer/index.html')

@app.route('/woer/Batting-01')
def woer_Batting_01():
    return render_template('woer/Batting-01.html')

@app.route('/woer/Batting-02')
def woer_Batting_02():
    return render_template('woer/Batting-02.html')

@app.route('/woer/Batting-03')
def woer_Batting_03():
    return render_template('woer/Batting-03.html')

@app.route('/woer/more-produce')
def more_produce():
    return render_template('woer/more-produce.html')

@app.route('/woer/intro-business')
def intro_business():
    return render_template('woer/intro-business.html')

@app.route('/woer/Cervical-curve-adjusting-pillow')
def Cervical_curve_adjusting_pillow():
    return render_template('woer/Cervical curve adjusting pillow.html')

@app.route('/woer/Luxury-beauty-pillow')
def Luxury_beauty_pillow():
    return render_template('woer/Luxury beauty pillow.html')

@app.route('/woer/noon-break-pillow')
def noon_break_pillow():
    return render_template('woer/noon-break-pillow.html')

@app.route('/woer/Pillow')
def Pillow():
    return render_template('woer/Pillow.html')

@app.route('/woer/pillow-for-winter-and-summer')
def pillow_for_winter_and_summer():
    return render_template('woer/pillow for winter and summer.html')

@app.route('/woer/Travel-baby-three-piece-set')
def Travel_baby_three_piece_set():
    return render_template('woer/Travel baby three piece set.html')

@app.route('/woer/Washable-pillow')
def Washable_pillow():
    return render_template('woer/Washable pillow.html')
if __name__ == "__main__":
    app.run(debug = True)
