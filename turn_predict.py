import plotly.graph_objects as go
import os
import copy
import cv2
import numpy as np
import math

vid_path = 'challenge.mp4'
vidObj = cv2.VideoCapture(vid_path)
success = 1
frames = []
rgb_frames = []

while success:
    success, rgb_image = vidObj.read()
    try:
        image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    except:
        assert len(frames) != 0, 'No frames found!!! Check video path'
        break
    frames.append(image)
    rgb_frames.append(rgb_image)
    


def homo_warp(img, h, w, inv=False):
    src_roi = np.float32([
      [595,450], #Top left corner
      [210,670], #BottomLeft corner            
      [1120,670], #Bottom Right corner
      [730,450] #TopRight corner
    ])
    dst_roi = np.float32([
      [0,0],
      [0,h],
      [w,h],
      [w,0]])
    #   [230,440], #TopLeft corner
    #   [230,670], #BottomLeft corner            
    #   [1120,670], #BottomRight corner
    #   [1120,440] #TopRight corner
    # ])

    H, _ = cv2.findHomography(src_roi, dst_roi)
    H_inv, _ = cv2.findHomography(dst_roi, src_roi)

    if inv:
      warped_frame = cv2.warpPerspective(img, H_inv, (w, h), flags=(cv2.INTER_LINEAR))
    else:
      warped_frame = cv2.warpPerspective(img, H, (w, h), flags=(cv2.INTER_LINEAR))

    return warped_frame

def generate_hist(bin_img):
    hist_plot = np.sum(bin_img, axis=0)
    return hist_plot


def do_sliding_window(bin_img, hist_plot, num_windows, min_pixels = 20, neighbrhud=150):
    global a_l, b_l, c_l,a_r, b_r, c_r 
    l_lane_param= np.empty(3)
    r_lane_param = np.empty(3)
    out_img = np.dstack((bin_img, bin_img, bin_img))*255

    center = int(len(hist_plot.tolist())/2)
    peak_l = np.argmax(hist_plot[:center])
    peak_r = center + np.argmax(hist_plot[center:])
    
    h_window = np.int(bin_img.shape[0]/num_windows)
    white = bin_img.nonzero()
    white_y = np.array(white[0])
    white_x = np.array(white[1])
    center_curr_window_l = peak_l
    center_curr_window_r = peak_r
    
    lane_idxs_l = []
    lane_idxs_r = []

    for window_idx in range(num_windows):
        win_bot = bin_img.shape[0] - (window_idx+1)*h_window
        win_top = bin_img.shape[0] - window_idx*h_window
        l_win_left = center_curr_window_l - neighbrhud
        l_win_right = center_curr_window_l + neighbrhud
        r_win_left = center_curr_window_r - neighbrhud
        r_win_right = center_curr_window_r + neighbrhud
        l_win_white = ((white_y >= win_bot) & (white_y < win_top) & (white_x >= l_win_left) & (white_x < l_win_right)).nonzero()[0]
        r_win_white = ((white_y >= win_bot) & (white_y < win_top) & (white_x >= r_win_left) & (white_x < r_win_right)).nonzero()[0]
        lane_idxs_l.append(l_win_white)
        lane_idxs_r.append(r_win_white)
        if len(l_win_white) > min_pixels:
            center_curr_window_l = np.int(np.mean(white_x[l_win_white]))
        if len(r_win_white) > min_pixels:        
            center_curr_window_r = np.int(np.mean(white_x[r_win_white]))

    lane_idxs_l = np.concatenate(lane_idxs_l)
    lane_idxs_r = np.concatenate(lane_idxs_r)
    left_line_x = white_x[lane_idxs_l]
    left_line_y = white_y[lane_idxs_l] 
    right_line_x = white_x[lane_idxs_r]
    right_line_y = white_y[lane_idxs_r] 
    #print(left_line_x.shape)

    # Fit a second order polynomial to each
    # If enough lane is not found, set a,b,c of the lanes as the average of their last 10 values
    if left_line_x.shape[0]>500:#try:
      fit_l = np.polyfit(left_line_y, left_line_x, 2)
      fit_r = np.polyfit(right_line_y, right_line_x, 2)
        
      a_l.append(fit_l[0])
      b_l.append(fit_l[1])
      c_l.append(fit_l[2])
      
      a_r.append(fit_r[0])
      b_r.append(fit_r[1])
      c_r.append(fit_r[2])
    
    else:#except:
      # print('came')
      # a_l_change = np.mean([a_l[i]-a_l[i-1] for i in range(-30, 0)])
      # b_l_change = np.mean([b_l[i]-b_l[i-1] for i in range(-30, 0)])
      # c_l_change = np.mean([c_l[i]-c_l[i-1] for i in range(-30, 0)])
      # a_r_change = np.mean([a_r[i]-a_r[i-1] for i in range(-30, 0)])
      # b_r_change = np.mean([b_r[i]-b_r[i-1] for i in range(-30, 0)])
      # c_r_change = np.mean([c_r[i]-c_r[i-1] for i in range(-30, 0)])

      a_l.append(np.mean(a_l[-10:]))#a_l[-1] + a_l_change)
      b_l.append(np.mean(b_l[-10:]))#b_l[-1] + b_l_change)
      c_l.append(np.mean(c_l[-10:]))#c_l[-1] + c_l_change)
      
      a_r.append(np.mean(a_r[-10:]))#a_r[-1] + a_r_change)
      b_r.append(np.mean(b_r[-10:]))#b_r[-1] + b_r_change)
      c_r.append(np.mean(c_r[-10:]))#c_r[-1] + c_r_change)
    
    l_lane_param[0] = np.mean(a_l[-10:])
    l_lane_param[1] = np.mean(b_l[-10:])
    l_lane_param[2] = np.mean(c_l[-10:])
    r_lane_param[0] = np.mean(a_r[-10:])
    r_lane_param[1] = np.mean(b_r[-10:])
    r_lane_param[2] = np.mean(c_r[-10:])    
    y_axis = np.linspace(0, bin_img.shape[0]-1, bin_img.shape[0])
    left_lane = l_lane_param[0]*y_axis**2 + l_lane_param[1]*y_axis + l_lane_param[2]
    right_lane = r_lane_param[0]*y_axis**2 + r_lane_param[1]*y_axis + r_lane_param[2]

    out_img[white_y[lane_idxs_l], white_x[lane_idxs_l]] = [255, 0, 100]
    out_img[white_y[lane_idxs_r], white_x[lane_idxs_r]] = [0, 100, 255]
    
    return out_img, left_lane, right_lane, l_lane_param, r_lane_param, y_axis

def calc_rad_curv(img, left_lane, right_lane):
    y_axis = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_max = np.max(y_axis)
    final_l_curve = np.polyfit(y_axis, left_lane, 2)
    final_r_curve = np.polyfit(y_axis, right_lane, 2)
    # Radii of curvature - based on formulas in the supplementary blog
    left_curve_ROC = ((1+(2*final_l_curve[0]*y_max+final_l_curve[1])**2)**1.5) / np.absolute(2*final_l_curve[0])
    right_curve_ROC = ((1+(2*final_r_curve[0]*y_max+final_r_curve[1])**2)**1.5) / np.absolute(2*final_r_curve[0])
    road_ROC = np.mean([left_curve_ROC, right_curve_ROC])
    return road_ROC

def generate_final_img(rgb_img, left_lane, right_lane):
    y_axis = np.linspace(0, rgb_img.shape[0]-1, rgb_img.shape[0])
    detected_lane = np.zeros_like(rgb_img)
    right = np.array([np.flipud(np.transpose(np.vstack([right_lane, y_axis])))])
    left = np.array([np.transpose(np.vstack([left_lane, y_axis]))])
    points = np.hstack((left, right))
    
    cv2.fillPoly(detected_lane, np.int_(points), (50,0,50))
    inv_warped_lane = homo_warp(detected_lane, h, w, inv=True)
    inv_warped_frame = cv2.addWeighted(rgb_img, 1, inv_warped_lane, 0.7, 0)
    return inv_warped_frame, inv_warped_lane

a_l, b_l, c_l = [],[],[]
a_r, b_r, c_r = [],[],[]

for frame_idx in range(len(frames)):
    img = frames[frame_idx]
    rgb_img = rgb_frames[frame_idx]
    h, w = img.shape
    warped_frame = homo_warp(img, h, w)
    _, bin_img = cv2.threshold(warped_frame, 195, 255, cv2.THRESH_BINARY)
    #cv2.imshow('warp_dividers', bin_img)
    
    hist_plot = generate_hist(bin_img)
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=np.array(range(0, w, 1)), y=hist_plot))
    # fig.show()
    window_op, left_lane, right_lane, l_lane_param, r_lane_param, y_axis = do_sliding_window(bin_img, hist_plot, num_windows=9)
    road_ROC = calc_rad_curv(bin_img, left_lane, right_lane)
    #print('Radius of Curvature:', road_ROC)
    inv_warped_bin = homo_warp(bin_img, h, w, inv=True)
    #cv2.imshow('inv_warped_dividers', inv_warped_bin)
    final_img, final_lane = generate_final_img(rgb_img, left_lane, right_lane)
    #cv2.imshow('final_lane', final_lane)
    #cv2.imshow('final_img', final_img)
    bottom_imgs = cv2.hconcat([bin_img, inv_warped_bin])
    bottom_imgs = cv2.cvtColor(bottom_imgs, cv2.COLOR_GRAY2BGR)
    bottom_imgs = cv2.resize(bottom_imgs, dsize=(int(bottom_imgs.shape[1]/2), bottom_imgs.shape[0]))
    
    top_img = cv2.vconcat([final_img, bottom_imgs])
    top_img = cv2.resize(top_img, dsize=(int(top_img.shape[1]*0.7), int(top_img.shape[0]*0.7)))
    cv2.imshow('output', top_img)
    if not os.path.exists('./results'):
        os.makedirs('./results')
    cv2.imwrite('results/turn{}.jpg'.format(str(frame_idx).zfill(3)), top_img)


    #cv2.imshow('window_op', window_op)
    cv2.waitKey(0)
    
    #break