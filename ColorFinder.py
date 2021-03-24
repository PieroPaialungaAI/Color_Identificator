import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn
import argparse
import scipy 
import seaborn as sns
from sklearn.cluster import KMeans
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import cv2

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def square_maker(image_path):
    image = get_image(image_path)
    h = int(image.shape[0])
    step_h = int(h/10) 
    w = int(image.shape[1])
    step_w = int(w/10) 
    X = np.arange(0,h+step_h,step_h)
    Y =np.arange(0,w+step_w,step_w)
    squares = [image[0:step_h,0:step_w]]
    for i in range(0,len(X)-1):
        for j in range(0,len(Y)-1):
            squares.append(image[X[i]:X[i+1],Y[j]:Y[j+1]])
    return np.array(squares)[1::]


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def ColorExtractor(image_path, number_of_colors):
    image = get_image(image_path)
    modified_image = image.reshape(image.shape[0]*image.shape[1], 3)
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    for i in range(len(rgb_colors)):
        rgb_colors[i] = rgb_colors[i].astype(int)
    return rgb_colors,hex_colors

def color_computing(image_path,rgb_colors):
    DIFF = []
    squared_image = square_maker(image_path)
    for square in squared_image:
        DIFF_COLOR = []
        for color in range(len(rgb_colors)):
            diff = np.abs(square - rgb_colors[color])
            DIFF_COLOR.append(diff.mean())
        DIFF.append(DIFF_COLOR)
    return np.array(DIFF)


def build_summary(image_path,rgb_colors,hex_colors):
    results = color_computing(image_path,rgb_colors)
    cols = ['Slice'] + hex_colors
    sorted_results = pd.DataFrame(columns= cols)
    k=0
    for r in results:
        d = {'Slice':int(k)}
        for c in range(len(hex_colors)):
            d[hex_colors[c]] = r[c]*100/r.sum()
        sorted_results = sorted_results.append(d,ignore_index=True)
        k=k+1
    return sorted_results

def color_to_s(summary,color,print_result=True):
    data_color = summary
    color_list = data_color.columns.to_list()[1::]
    colors = data_color[color_list]
    color_index = colors.columns.tolist().index(color)
    SLICE_NUMBER = []
    COLOR_V = [] 
    for i in range(len(colors)):
        arg_max = np.argmax(colors.loc[i])
        max_value = np.max(colors.loc[i])
        if color_index == arg_max :
            SLICE_NUMBER.append(i)
            COLOR_V.append(max_value)
    d = {'Slices':SLICE_NUMBER,'Percentage': COLOR_V}
    f = open("result.txt", "w")
    if print_result == True:
        for s in range(len(SLICE_NUMBER)):
            print('Slice selected : '+str(SLICE_NUMBER[s])+ ', Color percentage: %.2f'%(COLOR_V[s]))
            f.write('Slice selected : '+str(SLICE_NUMBER[s])+ ', Color percentage: %.2f \n  ' %(COLOR_V[s]))
    f.close()
    #    return d 
    

def main():
    
    args = parser.parse_args()
    im_path = args.image_path
    num_of_colors = args.color_number
    index_color = args.color
    palette = ColorExtractor(im_path,num_of_colors)
    data = build_summary(im_path, palette[0],palette[1])
    chosen_color = palette[1][index_color]
    color_to_s(data,chosen_color)
    
    
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_path',type=str,help = 'Path of the png image'),
    parser.add_argument('--color',type=int,help = 'Color Number from the palette'),
    parser.add_argument('--color_number',type=int,help = 'Number of colors')
    
    main()
    
