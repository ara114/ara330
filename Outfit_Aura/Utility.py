import random

imageMap = {}
files = ["dominant_colors(dress).csv", "dominant_colors(dress).csv", "dominant_colors(female-formal-tops).csv",
         "dominant_colors(female-informal-bottoms).csv", "dominant_colors(femal-informal-tops).csv", "dominant_colors(male-formal-bottoms).csv",
         "dominant_colors(male-formal-tops).csv", "dominant_colors(male-informal-bottoms).csv", "dominant_colors(male-informal-tops).csv"
         ]

def readColors():
    red_count = 0
    blue_count = 0
    black_count = 0
    white_count = 0
    for f in files:
        f = "colors/" + f
        input_file = open(f, "r")
        lines = input_file.readlines()
        id = 0
        for line in lines:
            if id == 0:
                id = id + 1
                continue
            datas = line.split(",")
            image_name = datas[0]
            rgb = datas[1].strip()
            rgb = rgb.replace("[", "")
            rgb = rgb.replace("]", "")
            irgbs = rgb.split(" ")
            #print(irgbs)
            rgbs = [0] * 3
            k = 0
            for entry in irgbs:
                if entry == '':
                    continue
                else:
                    rgbs[k] = entry
                    k = k+1
            if int(float(rgbs[0])) > int(float(rgbs[1])) and int(float(rgbs[0])) > int(float(rgbs[2])):
                imageMap[image_name] = 'red'
                red_count+=1
            elif int(float(rgbs[2])) > int(float(rgbs[1])) and int(float(rgbs[2])) > int(float(rgbs[0])):
                imageMap[image_name] = 'blue'
                blue_count+=1
            else:
                Y = 0.2126 * int(float(rgbs[0])) + 0.7152 * int(float(rgbs[2])) + 0.0722 * int(float(rgbs[2]))
                if Y < 128:
                    imageMap[image_name] = 'black'
                    black_count+=1
                else:
                    imageMap[image_name] = 'white'
                    white_count= white_count + 1
            id = id + 1
    #return (red_count,blue_count,black_count,white_count)
    return imageMap
"""
if __name__ == '__main__':
    R,BU,BL,WH = readColors()
    print("RED COUNT " + str(R))
    print("BLUE COUNT " + str(BU))
    print("BLACK COUNT " + str(BL))
    print("WHITE COUNT " + str(WH))
"""