#!/usr/bin/python3

#import bcrypt
import requests
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import mysql.connector
import os
import random

app = Flask(__name__)
app.secret_key = "secret_key"

from Utility import readColors

colorMap = readColors()

def filterByColor(imageList, color):
    selectImages = []
    for img in imageList:
        result = img.split("/")
        img_name = result[len(result) - 1]
        icolor = colorMap[img_name]
        print ("comparing color " + color + " with " + icolor + " for " + img)
        if icolor == color:
            selectImages.append(img)
    random.shuffle(selectImages)
    finalList= []

    m = 0
    for entry in selectImages:
        if m == 3:
            break
        finalList.append(entry)
        m= m+1
    return finalList


#starting page
@app.route('/')
def index():
    return  render_template('index.html')

#starting page
@app.route('/contact')
def contact():
    return  render_template('contact.html')

#starting page
@app.route('/matches')
def matches():
    return  render_template('matches.html')

def getFileList(dir_path) :
    res = []
    sub_path = dir_path
    dir_path = "C:/Users/ARA/Desktop/ara330/Outfit_Aura/static/images/" + dir_path
    print ("Directory path............." + dir_path)
    for file_path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, file_path)):
            res.append("/static/images/" + sub_path + "/" +file_path)
    print (res)
    return res

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['gender']
    color = request.form['color']
    type = request.form['type']
    category = request.form['category']

    session['selected_gender'] = gender
    session['selected_color'] = color
    session['selected_type'] = type
    session['selected_category'] = category
    result = []
    if gender == "male" :
        if type == "formal":
            if category == "top":
                result = getFileList("male/formal/tops")
            else:
                result = getFileList("male/formal/bottoms")
        else:
            if category == "top":
                result = getFileList("male/informal/tops")
            else:
                result = getFileList("male/informal/bottoms")
    else:
        if type == "formal":
            if category == "top":
                result = getFileList("female/formal/tops")
            else:
                result = getFileList("female/formal/bottoms")
        else:

            if category == "top":
                result = getFileList("female/informal/tops")
            elif category == "bottom":
                result = getFileList("female/informal/bottoms")
            else:
                result = getFileList("female/informal/dresses")
    count = len(result)
    #selected = random.randint(1, count)
    selected_images = []
    selected_images = filterByColor(result,color)
    """selected_images.append(result[0])
    selected_images.append(result[1])
    selected_images.append(result[2])"""
    #print ("Selected Image " + selected_images)
    if session['selected_category'] == 'dress':
        session['is_dress'] = 'yes'
        return  render_template('predict.html', selected_images= selected_images, is_dress='yes')
    else:
        session['is_dress'] = 'no'
        return render_template('predict.html', selected_images=selected_images, is_dress='no')

@app.route('/result', methods=['POST'])
def result():
    input_image = ""
    selected_images = request.form.getlist('selected_image')

    for name in selected_images:
        input_image = name
    #results = []
    session['first_item'] = input_image
    from resnet_pairs import getSuggestions
    results = getSuggestions("" +input_image, session['selected_type'], session['selected_gender'],session['selected_category'])
    print(results)
    if session['is_dress'] == 'yes':
        session['is_dress'] = 'no'
        return render_template('final.html', first_item=session['first_item'])
    else:
        return render_template('result.html', results=results)

@app.route('/merge', methods=['POST'])
def merge():
    input_image = ""
    selected_images = request.form.getlist('selected_image')
    for name in selected_images:
        input_image = name
    first_item = session['first_item']
    second_item = input_image
    print(first_item)
    print(second_item)
    return  render_template('final.html', first_item=first_item, second_item=second_item)

if __name__ == '__main__':
    app.run(debug=True)
