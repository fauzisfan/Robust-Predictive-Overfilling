# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:51:41 2020

@author: wnlab
"""

import numpy as np
import csv as csv
import pandas as pd
import matplotlib.pyplot as plt
import quaternion
from rms import rms
import math

def eul2quat_bio (eul):
	# Convert Euler [oculus] to Quaternion [oculus]
	eul = np.deg2rad(eul)
	X_eul = eul[:,0]
	Y_eul = eul[:,1]
	Z_eul = eul[:,2]

	cos_pitch, sin_pitch = np.cos(X_eul/2), np.sin(X_eul/2)
	cos_yaw, sin_yaw = np.cos(Y_eul/2), np.sin(Y_eul/2)
	cos_roll, sin_roll = np.cos(Z_eul/2), np.sin(Z_eul/2)

	# order: w,x,y,z
	# quat = unit_q(quat)
	quat = np.nan * np.ones( (eul.shape[0],4) )
	quat[:,3] = cos_pitch * cos_yaw * cos_roll + sin_pitch * sin_yaw * sin_roll
	quat[:,0] = cos_pitch * cos_yaw * sin_roll - sin_pitch * sin_yaw * cos_roll
	quat[:,1] = cos_pitch * sin_yaw * cos_roll + sin_pitch * cos_yaw * sin_roll 
	quat[:,2] = sin_pitch * cos_yaw * cos_roll - cos_pitch * sin_yaw * sin_roll		
	return (quat)

def predict_overfilling(predict_orientation, predict_orientation_min, predict_orientation_max, hmd_projection):
	# calc_optimal_overhead(input_orientation, predict_orientation, input_projection)
	q_d_min = np.matmul(
			np.linalg.inv(quaternion.as_rotation_matrix(predict_orientation)),
			quaternion.as_rotation_matrix(predict_orientation_min)
			)
	q_d_max = np.matmul(
			np.linalg.inv(quaternion.as_rotation_matrix(predict_orientation)),
			quaternion.as_rotation_matrix(predict_orientation_max)
			)
	
	lt_min = np.matmul(q_d_min, [hmd_projection[0], hmd_projection[1], 1])
	p_lt_min = np.dot(lt_min, 1 / lt_min[2])
	
	rt_min = np.matmul(q_d_min, [hmd_projection[2], hmd_projection[1], 1])
	p_rt_min = np.dot(rt_min, 1 / rt_min[2])
	
	rb_min = np.matmul(q_d_min, [hmd_projection[2], hmd_projection[3], 1])
	p_rb_min = np.dot(rb_min, 1 / rb_min[2])
	
	lb_min = np.matmul(q_d_min, [hmd_projection[0], hmd_projection[3], 1])
	p_lb_min = np.dot(lb_min, 1 / lb_min[2])
	
	lt_max = np.matmul(q_d_max, [hmd_projection[0], hmd_projection[1], 1])
	p_lt_max = np.dot(lt_max, 1 / lt_max[2])
	
	rt_max = np.matmul(q_d_max, [hmd_projection[2], hmd_projection[1], 1])
	p_rt_max = np.dot(rt_max, 1 / rt_max[2])
	
	rb_max = np.matmul(q_d_max, [hmd_projection[2], hmd_projection[3], 1])
	p_rb_max = np.dot(rb_max, 1 / rb_max[2])
	
	lb_max = np.matmul(q_d_max, [hmd_projection[0], hmd_projection[3], 1])
	p_lb_max = np.dot(lb_max, 1 / lb_max[2])
	
	p_l1 = min(p_lt_min[0], p_rt_min[0], p_rb_min[0], p_lb_min[0])
	p_l2 = min(p_lt_max[0], p_rt_max[0], p_rb_max[0], p_lb_max[0])
	p_t1 = max(p_lt_min[1], p_rt_min[1], p_rb_min[1], p_lb_min[1])
	p_t2 = max(p_lt_max[1], p_rt_max[1], p_rb_max[1], p_lb_max[1])
	p_r1 = max(p_lt_min[0], p_rt_min[0], p_rb_min[0], p_lb_min[0])
	p_r2 = max(p_lt_max[0], p_rt_max[0], p_rb_max[0], p_lb_max[0])
	p_b1 = min(p_lt_min[1], p_rt_min[1], p_rb_min[1], p_lb_min[1])
	p_b2 = min(p_lt_max[1], p_rt_max[1], p_rb_max[1], p_lb_max[1])
	
	p_l = (p_l1+p_l2)/2
	p_t = (p_t1+p_t2)/2
	p_r = (p_r1+p_r2)/2
	p_b = (p_b1+p_b2)/2
	
	size = max(p_r - p_l, p_t - p_b)
	a_overfilling = size * size
	
	a_hmd = (hmd_projection[2] - hmd_projection[0]) * (hmd_projection[1] - hmd_projection[3])
	
	margins = np.max(np.abs([p_l, p_t, p_r, p_b]))
#	return (a_overfilling / a_hmd - 1)*100, [-margins, margins, margins, -margins]
#	return [-margins, margins, margins, -margins]
	return [-np.abs(p_l),np.abs(p_t),np.abs(p_r),-np.abs(p_b)]

def calc_optimal_overhead(hmd_orientation, frame_orientation, hmd_projection):
	# calc_optimal_overhead(input_orientation, predict_orientation, input_projection)
	q_d = np.matmul(
			np.linalg.inv(quaternion.as_rotation_matrix(hmd_orientation)),
			quaternion.as_rotation_matrix(frame_orientation)
			)
	#Projection Orientation:
		#hmd_projection[0] : Left (Negative X axis)
		#hmd_projection[1] : Top (Positive Y axis)
		#hmd_projection[2] : Right (Positive X axis)
		#hmd_projection[3] : Bottom (Negative Y axis)
		
	lt = np.matmul(q_d, [hmd_projection[0], hmd_projection[1], 1])
	p_lt = np.dot(lt, 1 / lt[2])
	
	rt = np.matmul(q_d, [hmd_projection[2], hmd_projection[1], 1])
	p_rt = np.dot(rt, 1 / rt[2])
	
	rb = np.matmul(q_d, [hmd_projection[2], hmd_projection[3], 1])
	p_rb = np.dot(rb, 1 / rb[2])
	
	lb = np.matmul(q_d, [hmd_projection[0], hmd_projection[3], 1])
	p_lb = np.dot(lb, 1 / lb[2])
	
	p_l = min(p_lt[0], p_rt[0], p_rb[0], p_lb[0])
	p_t = max(p_lt[1], p_rt[1], p_rb[1], p_lb[1])
	p_r = max(p_lt[0], p_rt[0], p_rb[0], p_lb[0])
	p_b = min(p_lt[1], p_rt[1], p_rb[1], p_lb[1])
	
	size = max(p_r - p_l, p_t - p_b)
	a_overfilling = size * size
	
	a_hmd = (hmd_projection[2] - hmd_projection[0]) * (hmd_projection[1] - hmd_projection[3])
	
	margins = np.max(np.abs([p_l, p_t, p_r, p_b]))
#	return (a_overfilling / a_hmd - 1)*100, [-margins, margins, margins, -margins]
#	return [-margins, margins, margins, -margins]
	return [-np.abs(p_l),np.abs(p_t),np.abs(p_r),-np.abs(p_b)]

def predictive_overfilling(input_orientation, prediction, input_projection):
	IPD = input_projection[2]-input_projection[0]
	h = (input_projection[1]-input_projection[3])/2
	r = np.sqrt(h**2+1/4*IPD**2)
	
	input_angle = np.arctan(IPD/(2*h))
	pitch_diff = prediction[0]-input_orientation[0]
	roll_diff = prediction[1]-input_orientation[1]
	yaw_diff = prediction[2]-input_orientation[2]
	
	if(yaw_diff<0):
		x_r = r*np.sin(input_angle-yaw_diff)
		x_l = min(input_projection[0],-r*np.sin(input_angle+yaw_diff))
	else:
		x_r = max(input_projection[2],r*np.sin(input_angle-yaw_diff))
		x_l = -r*np.sin(input_angle+yaw_diff)
	
	if(pitch_diff<0):
		x_t = r*np.sin(input_angle+pitch_diff)
		x_b = min(input_projection[3],-r*np.sin(input_angle+pitch_diff))
	else:
		x_t = max(input_projection[2],r*np.sin(input_angle+pitch_diff))
		x_b = -r*np.sin(input_angle+pitch_diff)
		
	if (roll_diff<0):
		x_rr = r*np.sin(input_angle-roll_diff)
		x_ll = -r*np.sin(input_angle-roll_diff)
		x_tt = r*np.cos(input_angle+roll_diff)
		x_bb = -r*np.cos(input_angle+roll_diff)
	else:
		x_rr = r*np.sin(input_angle+roll_diff)
		x_ll = -r*np.sin(input_angle+roll_diff)
		x_tt = r*np.cos(input_angle-roll_diff)
		x_bb = -r*np.cos(input_angle-roll_diff)
	
	p_r = x_r+x_rr-IPD/2
	p_l = x_l+x_ll+IPD/2
	p_t = x_t+x_tt-IPD/2
	p_b = x_b+x_bb+IPD/2
	
	return [-np.abs(p_l),np.abs(p_t),np.abs(p_r),-np.abs(p_b)]
	
# File name of the trace file
inFile = "s20200915_scene(3)_user(1)_(300).csv"
outFile1 = "20200915_scene(3)_user(1)_(300)_optimal_new.csv"
outFile2 = "20200915_scene(3)_user(1)_(300)_predictive_signed.csv"
outFile3 = "20200915_scene(3)_user(1)_(300)_robust_predictive.csv"

anticipation_time = 300
anticipation_size = 18

# read csv
try:
	df_tracefile = pd.read_csv(inFile)
	timestamp = np.array(df_tracefile[['timestamp']], dtype=np.float32)
	bios_data = np.array(df_tracefile[['biosignal_0','biosignal_1','biosignal_2','biosignal_3','biosignal_4','biosignal_5','biosignal_6','biosignal_7']], dtype=np.float32)
	gyro_data = np.array(df_tracefile[['angular_vec_x', 'angular_vec_y', 'angular_vec_z']], dtype=np.float32)
	acce_data = np.array(df_tracefile[['acceleration_x', 'acceleration_y', 'acceleration_z']], dtype=np.float32)
	magn_data = np.array(df_tracefile[['magnetic_x', 'magnetic_y', 'magnetic_z']], dtype=np.float32)
	orie_data = np.array(df_tracefile[['input_orientation_pitch', 'input_orientation_roll', 'input_orientation_yaw']], dtype=np.float32)
	quat_data = np.array(df_tracefile[['input_orientation_x', 'input_orientation_y', 'input_orientation_z', 'input_orientation_w']], dtype=np.float32)
	proj_data = np.array(df_tracefile[['input_projection_left', 'input_projection_top', 'input_projection_right', 'input_projection_bottom']], dtype=np.float32)
	pred_time = np.array(df_tracefile[['prediction_time']], dtype=np.float32)
	pred_orie_data = np.array(df_tracefile[['predicted_orientation_pitch', 'predicted_orientation_roll', 'predicted_orientation_yaw']], dtype=np.float32)
	pred_quat_data = np.array(df_tracefile[['predicted_orientation_x', 'predicted_orientation_y', 'predicted_orientation_z', 'predicted_orientation_w']], dtype=np.float32)
	pred_proj_data = np.array(df_tracefile[['predicted_projection_left', 'predicted_projection_top', 'predicted_projection_right', 'predicted_projection_bottom']], dtype=np.float32)
except:
	raise
	
#get only the test part
timestamp = np.split(timestamp,2)[1]
bios_data = np.split(bios_data,2)[1]
gyro_data = np.split(gyro_data,2)[1]
acce_data = np.split(acce_data,2)[1]
magn_data = np.split(magn_data,2)[1]
orie_data = np.split(orie_data,2)[1]
quat_data = np.split(quat_data,2)[1]
proj_data = np.split(proj_data,2)[1]
pred_time = np.split(pred_time,2)[1]
pred_orie_data = np.split(pred_orie_data,2)[1]
pred_proj_data = np.split(pred_proj_data,2)[1]
pred_quat_data = np.split(pred_quat_data,2)[1]

#MAE Error Calculation
# convert to degree first
pred_orie_data_deg = np.rad2deg(pred_orie_data)
orie_data_deg = np.rad2deg(orie_data)
signed_pred_error = pred_orie_data_deg - orie_data_deg
MAE_error = np.nanmean(np.abs(signed_pred_error),axis=0)
print("MAE Error: Pitch, Roll, Yaw:")
print(MAE_error)

## plot error
#plt.figure()
#plt.plot((timestamp-timestamp[0])/705600000, signed_pred_error[:, 0], linewidth=1)
#plt.plot((timestamp-timestamp[0])/705600000, signed_pred_error[:, 1], linewidth=1)
#plt.plot((timestamp-timestamp[0])/705600000, signed_pred_error[:, 2], linewidth=1)
#plt.legend(['Pitch','Roll','Yaw'])
#plt.title('Error (Signed) Anticipation Time: ' + str(anticipation_time) + ' ms')
#plt.grid()
#plt.xlabel('Time (s)')
#plt.ylabel('Error (deg)')

# calculate ellipsoid radius for errors
signed_pred_error_pitch = signed_pred_error[:, 0]; # pitch
signed_pred_error_roll = signed_pred_error[:, 1]; # roll
signed_pred_error_yaw = signed_pred_error[:, 2]; # yaw

np.savetxt("signed_pred_error.csv", signed_pred_error, delimiter=",")

k = 50
tracked_std_pitch = np.zeros(len(signed_pred_error_pitch))
tracked_std_roll = np.zeros(len(signed_pred_error_pitch))
tracked_std_yaw = np.zeros(len(signed_pred_error_pitch))
tracked_mean_pitch = np.zeros(len(signed_pred_error_pitch))
tracked_mean_roll = np.zeros(len(signed_pred_error_pitch))
tracked_mean_yaw = np.zeros(len(signed_pred_error_pitch))

for i in range(0,len(signed_pred_error_pitch)):
	# create empty array
#	tracked_array_pitch = np.array([]);
#	tracked_array_roll = np.array([]);
#	tracked_array_yaw = np.array([]);
	if i<k:
#		for j in range(i,-1,-1):
#			tracked_array_pitch = np.append(tracked_array_pitch, signed_pred_error_pitch[j])
#			tracked_array_roll = np.append(tracked_array_roll, signed_pred_error_roll[j])
#			tracked_array_yaw = np.append(tracked_array_yaw, signed_pred_error_yaw[j])
		
#		more simple way
		tracked_array_pitch = signed_pred_error_pitch[:i+1]
		tracked_array_roll = signed_pred_error_roll[:i+1]
		tracked_array_yaw = signed_pred_error_yaw[:i+1]
	else:
#		for j in range(i,i-k,-1):
#			tracked_array_pitch = np.append(tracked_array_pitch, signed_pred_error_pitch[j])
#			tracked_array_roll = np.append(tracked_array_roll, signed_pred_error_roll[j])
#			tracked_array_yaw = np.append(tracked_array_yaw, signed_pred_error_yaw[j])
        
#		more simple way
		tracked_array_pitch = signed_pred_error_pitch[i-k:i+1]
		tracked_array_roll = signed_pred_error_roll[i-k:i+1]
		tracked_array_yaw = signed_pred_error_yaw[i-k:i+1]
	
	tracked_std_pitch[i] = np.std(tracked_array_pitch)
	tracked_std_roll[i] = np.std(tracked_array_roll)
	tracked_std_yaw[i] = np.std(tracked_array_yaw)
	
	tracked_mean_pitch[i] = np.mean(tracked_array_pitch)
	tracked_mean_roll[i] = np.mean(tracked_array_roll)
	tracked_mean_yaw[i] = np.mean(tracked_array_yaw)

d95 = np.sqrt(-2*np.log(1-0.95))
d99 = np.sqrt(-2*np.log(1-0.99))

ellipsoid_radius_pitch_95 = tracked_std_pitch * d95
ellipsoid_radius_roll_95 = tracked_std_roll * d95
ellipsoid_radius_yaw_95 = tracked_std_yaw * d95

ellipsoid_radius_pitch_99 = tracked_mean_pitch * d99
ellipsoid_radius_roll_99 = tracked_mean_roll * d99
ellipsoid_radius_yaw_99 = tracked_mean_yaw * d99

radius_pitch 	= ellipsoid_radius_pitch_99[-1]
radius_yaw 		= ellipsoid_radius_yaw_99[-1]
radius_pitch2 	= ellipsoid_radius_pitch_95[-1]
radius_yaw2 	= ellipsoid_radius_yaw_95[-1]

#this is when true mean and std
#radius_pitch = np.std(signed_pred_error_pitch) * d99
#radius_yaw = np.std(signed_pred_error_yaw) * d99
#radius_pitch2 = np.std(signed_pred_error_pitch) * d95
#radius_yaw2 = np.std(signed_pred_error_yaw) * d95

''' Draw Ellipsoid '''

#x_ellipse = np.arange(-radius_yaw,radius_yaw,0.0001)
#x_ellipse_squared = np.square(x_ellipse)
#y_ellipse_plus = (radius_pitch/radius_yaw)*np.sqrt(-x_ellipse_squared+radius_yaw*radius_yaw)
#y_ellipse_minus = -(radius_pitch/radius_yaw)*np.sqrt(-x_ellipse_squared+radius_yaw*radius_yaw)
#x_ellipse2 = np.arange(-radius_yaw2,radius_yaw2,0.0001)b 
#x_ellipse_squared2 = np.square(x_ellipse2)
#y_ellipse_plus2 = (radius_pitch2/radius_yaw2)*np.sqrt(-x_ellipse_squared2+radius_yaw2*radius_yaw2)
#y_ellipse_minus2 = -(radius_pitch2/radius_yaw2)*np.sqrt(-x_ellipse_squared2+radius_yaw2*radius_yaw2)

# plt.figure()
# plt.scatter(signed_pred_error_yaw, signed_pred_error_pitch, color='blue')

# plt.scatter(x_ellipse, y_ellipse_plus, color='red',marker='.', linewidth=0.1)
# plt.scatter(x_ellipse, y_ellipse_minus, color='red',marker='.', linewidth=0.1)

# plt.scatter(x_ellipse2, y_ellipse_plus2, color='green',marker='.', linewidth=0.1)
# plt.scatter(x_ellipse2, y_ellipse_minus2, color='green',marker='.', linewidth=0.1)

# plt.title("Scatter Plot of Yaw vs Pitch")
# plt.xlabel("Yaw Error")
# plt.ylabel("Pitch Error")
# plt.grid()
# plt.axhline(0, color='black')
# plt.axvline(0, color='black')
# plt.xlim(-30,30)
# plt.ylim(-30,30)

euler_pred_ann_pitch = pred_orie_data_deg[:,0]
euler_pred_ann_roll = pred_orie_data_deg[:,1]
euler_pred_ann_yaw = pred_orie_data_deg[:,2]

euler_pred_ann_pitch_min = euler_pred_ann_pitch - ellipsoid_radius_pitch_99
euler_pred_ann_roll_min = euler_pred_ann_roll - ellipsoid_radius_roll_99
euler_pred_ann_yaw_min = euler_pred_ann_yaw - ellipsoid_radius_yaw_99

euler_pred_ann_pitch_min95 = euler_pred_ann_pitch - ellipsoid_radius_pitch_95
euler_pred_ann_roll_min95 = euler_pred_ann_roll - ellipsoid_radius_roll_95
euler_pred_ann_yaw_min95 = euler_pred_ann_yaw - ellipsoid_radius_yaw_95

euler_pred_ann_pitch_max = euler_pred_ann_pitch + ellipsoid_radius_pitch_99
euler_pred_ann_roll_max = euler_pred_ann_roll + ellipsoid_radius_roll_99
euler_pred_ann_yaw_max = euler_pred_ann_yaw + ellipsoid_radius_yaw_99

euler_pred_ann_pitch_max95 = euler_pred_ann_pitch + ellipsoid_radius_pitch_95
euler_pred_ann_roll_max95 = euler_pred_ann_roll + ellipsoid_radius_roll_95
euler_pred_ann_yaw_max95 = euler_pred_ann_yaw + ellipsoid_radius_yaw_95

euler_pred_ann_test_min = np.column_stack([euler_pred_ann_pitch_min, euler_pred_ann_roll_min, euler_pred_ann_yaw_min])
euler_pred_ann_test_max = np.column_stack([euler_pred_ann_pitch_max, euler_pred_ann_roll_max, euler_pred_ann_yaw_max])

euler_pred_ann_test_min95 = np.column_stack([euler_pred_ann_pitch_min95, euler_pred_ann_roll_min95, euler_pred_ann_yaw_min95])
euler_pred_ann_test_max95 = np.column_stack([euler_pred_ann_pitch_max95, euler_pred_ann_roll_max95, euler_pred_ann_yaw_max95])

# convert to quaternions
quat_pred_ann_test_ = eul2quat_bio(pred_orie_data_deg) 
quat_pred_ann_test_min_ = eul2quat_bio(euler_pred_ann_test_min)
quat_pred_ann_test_max_ = eul2quat_bio(euler_pred_ann_test_max)

quat_pred_ann_test_min95_ = eul2quat_bio(euler_pred_ann_test_min95)
quat_pred_ann_test_max95_ = eul2quat_bio(euler_pred_ann_test_max95)

#quat_pred_ann_test = quat_pred_ann_test_[anticipation_size:]
#quat_pred_ann_test_min = quat_pred_ann_test_min_[:-anticipation_size]
#quat_pred_ann_test_max = quat_pred_ann_test_max_[:-anticipation_size]
#
#quat_pred_ann_test_min95 = quat_pred_ann_test_min95_[:-anticipation_size]
#quat_pred_ann_test_max95 = quat_pred_ann_test_max95_[:-anticipation_size]

idx1 = [2,3,0,1]
quat_pred_ann_test = quat_pred_ann_test_[:-anticipation_size:,idx1]
quat_pred_ann_test_min = quat_pred_ann_test_min_[anticipation_size:,idx1]
quat_pred_ann_test_max = quat_pred_ann_test_max_[anticipation_size:,idx1]

quat_pred_ann_test_min95 = quat_pred_ann_test_min95_[anticipation_size:,idx1]
quat_pred_ann_test_max95 = quat_pred_ann_test_max95_[anticipation_size:,idx1]

#Possible tracked pred
tracked_radius_pitch = d99*tracked_mean_pitch
tracked_radius_roll = d99*tracked_mean_roll
tracked_radius_yaw = d99*tracked_mean_yaw

euler_pos_pred_pitch = euler_pred_ann_pitch - tracked_radius_pitch
euler_pos_pred_roll = euler_pred_ann_roll - tracked_radius_roll
euler_pos_pred_yaw = euler_pred_ann_yaw - tracked_radius_yaw

euler_pos_pred = np.column_stack([euler_pos_pred_pitch, euler_pos_pred_roll, euler_pos_pred_yaw])
quat_pos_pred = eul2quat_bio(euler_pos_pred)

quat_pos_pred = quat_pos_pred[anticipation_size:,idx1]

## Calculate initial predicted values
pred_values = np.empty((0,4), dtype=np.float32)
pred_overhead = np.empty((0,1), dtype=np.float32)
for i in range(0, len(quat_pred_ann_test)+anticipation_size):
	if (i<len(quat_pred_ann_test)):
#		np_quat_pred_ann_test = np.quaternion(quat_pred_ann_test[i][0], quat_pred_ann_test[i][1], quat_pred_ann_test[i][2], quat_pred_ann_test[i][3])
		np_quat_pred_ann_test = np.quaternion(quat_data[i,0],quat_data[i,3],quat_data[i,1],quat_data[i,2])

#		np_quat_pred_ann_test_min = np.quaternion(quat_pred_ann_test_min[i][0], quat_pred_ann_test_min[i][1], quat_pred_ann_test_min[i][2], quat_pred_ann_test_min[i][3])
#		np_quat_pred_ann_test_max = np.quaternion(quat_pred_ann_test_max[i][0], quat_pred_ann_test_max[i][1], quat_pred_ann_test_max[i][2], quat_pred_ann_test_max[i][3])
#		
#		np_quat_pred_ann_test_min95 = np.quaternion(quat_pred_ann_test_min95[i][0], quat_pred_ann_test_min95[i][1], quat_pred_ann_test_min95[i][2], quat_pred_ann_test_min95[i][3])
#		np_quat_pred_ann_test_max95 = np.quaternion(quat_pred_ann_test_max95[i][0], quat_pred_ann_test_max95[i][1], quat_pred_ann_test_max95[i][2], quat_pred_ann_test_max95[i][3])
#		
#		pred_overhead_values = predict_overfilling(np_quat_pred_ann_test, np_quat_pred_ann_test_min, np_quat_pred_ann_test_max, np.array([-1,1,1,-1]))
#		pred_values = np.vstack((pred_values,pred_overhead_values))

		np_possible_predict_orientation = np.quaternion(quat_pos_pred[i,0], quat_pos_pred[i,1], quat_pos_pred[i,2], quat_pos_pred[i,3])
		pred_overhead_values = calc_optimal_overhead(np_quat_pred_ann_test, np_possible_predict_orientation, np.array([-1,1,1,-1]))
		pred_values = np.vstack((pred_values,pred_overhead_values))
	else:
		pred_values = np.vstack((pred_values,np.array([-1,1,1,-1])))
        
#print('Prediction Overhead : {:.2f}%'.format(np.nanmean(np.abs(pred_overhead))))
#rewrite CSV
raw_pred_data = np.column_stack(
	[timestamp, 
	bios_data, 
	gyro_data, 
	acce_data, 
	magn_data, 
	quat_data,
	orie_data, 
	proj_data,
	pred_time,
	pred_quat_data, # use pred_quat_data for predicted
	pred_orie_data, # use pred_orie_data for predicted
	pred_values
	]
)

## Calculate optimal values
#idx = [1,2,0,3]
idx = [0,3,1,2]

hmd_orientation_ = quat_data[:,idx]

hmd_orientation = hmd_orientation_[:-anticipation_size]			# current quaternion orientation
frame_orientation = hmd_orientation_[anticipation_size:]		# predicted quaternion orientation, obtained (anticipation_time) before

optimal_values = np.empty((0,4), dtype=np.float32)
opt_overhead = np.empty((0,1), dtype=np.float32)

for i in range(0, len(hmd_orientation)+anticipation_size):
	if (i<len(hmd_orientation)):
	#if (ann_rms_stream_test[i] < ann_rms_99):
		input_orientation = np.quaternion(
			hmd_orientation[i,0],
			hmd_orientation[i,1],
			hmd_orientation[i,2],
			hmd_orientation[i,3]
			)
		predict_orientation = np.quaternion(
			frame_orientation[i,0],
			frame_orientation[i,1],
			frame_orientation[i,2],
			frame_orientation[i,3]
			)
		input_projection = [
			proj_data[i,0],
			proj_data[i,1],
			proj_data[i,2],
			proj_data[i,3]
			]

		optimal_overhead_values = calc_optimal_overhead(input_orientation, predict_orientation, input_projection)
		optimal_values = np.vstack((optimal_values,optimal_overhead_values))
	else:
		optimal_values = np.vstack((optimal_values,np.array([-1,1,1,-1])))

#print('Optimal Overhead : {:.2f}%'.format(np.nanmean(np.abs(opt_overhead))))
        
#rewrite CSV
raw_optimal_data = np.column_stack(
	[timestamp, 
	bios_data, 
	gyro_data, 
	acce_data, 
	magn_data, 
	quat_data,
	orie_data, 
	proj_data,
	pred_time,
	# because this is optimal, we use actual data as we already know the actual position in the future
	quat_data, # use pred_quat_data for predicted
	orie_data, # use pred_orie_data for predicted
	optimal_values
	]
)

input_orientation = orie_data[:-anticipation_size]
prediction = pred_orie_data[anticipation_size:]	# current quaternion orientation
input_projection = proj_data		# predicted quaternion orientation, obtained (anticipation_time) before

predictive_values = np.empty((0,4), dtype=np.float32)

for i in range(0, len(input_orientation)+anticipation_size):
	if (i<len(input_orientation)):

		overhead_val = predictive_overfilling(input_orientation[i], prediction[i], input_projection[i])
		predictive_values = np.vstack((predictive_values,overhead_val))
	else:
		predictive_values = np.vstack((predictive_values, input_projection[i]))

raw_predictive_data = np.column_stack(
	[timestamp, 
	bios_data, 
	gyro_data, 
	acce_data, 
	magn_data, 
	quat_data,
	orie_data, 
	proj_data,
	pred_time,
	# because this is optimal, we use actual data as we already know the actual position in the future
	quat_data, # use pred_quat_data for predicted
	orie_data, # use pred_orie_data for predicted
	predictive_values
	]
)

# write the optimal into file
df = pd.DataFrame(raw_optimal_data,columns = ['timestamp',
							'biosignal_0', 'biosignal_1', 'biosignal_2', 'biosignal_3', 'biosignal_4', 'biosignal_5', 'biosignal_6', 'biosignal_7',
							'angular_vec_x', 'angular_vec_y', 'angular_vec_z',
							'acceleration_x', 'acceleration_y', 'acceleration_z',
							'magnetic_x', 'magnetic_y', 'magnetic_z',
							'input_orientation_x', 'input_orientation_y', 'input_orientation_z', 'input_orientation_w',
							'input_orientation_pitch', 'input_orientation_roll', 'input_orientation_yaw',
							'input_projection_left', 'input_projection_top', 'input_projection_right', 'input_projection_bottom',
							'prediction_time',
							'predicted_orientation_x', 'predicted_orientation_y', 'predicted_orientation_z', 'predicted_orientation_w',
							'predicted_orientation_pitch', 'predicted_orientation_roll', 'predicted_orientation_yaw',
							'predicted_projection_left', 'predicted_projection_top', 'predicted_projection_right', 'predicted_projection_bottom',
							])
export_csv = df.to_csv (str(outFile1), index = None, header=True)

# write the predicted one into file 
df = pd.DataFrame(raw_pred_data,columns = ['timestamp',
							'biosignal_0', 'biosignal_1', 'biosignal_2', 'biosignal_3', 'biosignal_4', 'biosignal_5', 'biosignal_6', 'biosignal_7',
							'angular_vec_x', 'angular_vec_y', 'angular_vec_z',
							'acceleration_x', 'acceleration_y', 'acceleration_z',
							'magnetic_x', 'magnetic_y', 'magnetic_z',
							'input_orientation_x', 'input_orientation_y', 'input_orientation_z', 'input_orientation_w',
							'input_orientation_yaw', 'input_orientation_pitch', 'input_orientation_roll',
							'input_projection_left', 'input_projection_top', 'input_projection_right', 'input_projection_bottom',
							'prediction_time',
							'predicted_orientation_x', 'predicted_orientation_y', 'predicted_orientation_z', 'predicted_orientation_w',
							'predicted_orientation_yaw', 'predicted_orientation_pitch', 'predicted_orientation_roll',
							'predicted_projection_left', 'predicted_projection_top', 'predicted_projection_right', 'predicted_projection_bottom',
							])
export_csv = df.to_csv (str(outFile2), index = None, header=True)

# write the predicted one into file 
df = pd.DataFrame(raw_predictive_data,columns = ['timestamp',
							'biosignal_0', 'biosignal_1', 'biosignal_2', 'biosignal_3', 'biosignal_4', 'biosignal_5', 'biosignal_6', 'biosignal_7',
							'angular_vec_x', 'angular_vec_y', 'angular_vec_z',
							'acceleration_x', 'acceleration_y', 'acceleration_z',
							'magnetic_x', 'magnetic_y', 'magnetic_z',
							'input_orientation_x', 'input_orientation_y', 'input_orientation_z', 'input_orientation_w',
							'input_orientation_yaw', 'input_orientation_pitch', 'input_orientation_roll',
							'input_projection_left', 'input_projection_top', 'input_projection_right', 'input_projection_bottom',
							'prediction_time',
							'predicted_orientation_x', 'predicted_orientation_y', 'predicted_orientation_z', 'predicted_orientation_w',
							'predicted_orientation_yaw', 'predicted_orientation_pitch', 'predicted_orientation_roll',
							'predicted_projection_left', 'predicted_projection_top', 'predicted_projection_right', 'predicted_projection_bottom',
							])
export_csv = df.to_csv (str(outFile3), index = None, header=True)

## Check
#print(optimal_values)
#print(pred_values)

# plot margins just to check
plt.figure()
plt.plot((timestamp-timestamp[0])/705600000, abs(optimal_values[:, 0])-1, linewidth=1)
plt.plot((timestamp-timestamp[0])/705600000, abs(pred_values[:, 0])-1, linewidth=1)
plt.plot((timestamp-timestamp[0])/705600000, abs(predictive_values[:, 0])-1, linewidth=1)
plt.legend(['Optimal','Predictive','Robust'])
plt.title('Margins (Left) Comparison')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Margins')

# plot margins just to check
plt.figure()
plt.plot((timestamp-timestamp[0])/705600000, abs(optimal_values[:, 1])-1, linewidth=1)
plt.plot((timestamp-timestamp[0])/705600000, abs(pred_values[:, 1])-1, linewidth=1)
plt.plot((timestamp-timestamp[0])/705600000, abs(predictive_values[:, 1])-1, linewidth=1)
plt.legend(['Optimal','Predictive','Robust'])
plt.title('Margins (Top) Comparison')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Margins')

# plot margins just to check
plt.figure()
plt.plot((timestamp-timestamp[0])/705600000, abs(optimal_values[:, 2])-1, linewidth=1)
plt.plot((timestamp-timestamp[0])/705600000, abs(pred_values[:, 2])-1, linewidth=1)
plt.plot((timestamp-timestamp[0])/705600000, abs(predictive_values[:, 2])-1, linewidth=1)
plt.legend(['Optimal','Predictive','Robust'])
plt.title('Margins (Right) Comparison')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Margins')

# plot margins just to check
plt.figure()
plt.plot((timestamp-timestamp[0])/705600000, abs(optimal_values[:, 3])-1, linewidth=1)
plt.plot((timestamp-timestamp[0])/705600000, abs(pred_values[:, 3])-1, linewidth=1)
plt.plot((timestamp-timestamp[0])/705600000, abs(predictive_values[:, 3])-1, linewidth=1)
plt.legend(['Optimal','Predictive','Robust'])
plt.title('Margins (Bottom) Comparison')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Margins')

plt.show()

overfill_mae_test = np.nanmean(np.abs(abs(pred_values)-abs(optimal_values)), axis=0)
rms_stream_test = np.apply_along_axis(rms,1,np.abs(abs(pred_values)-abs(optimal_values)))
overfill_rms_test = np.nanmean(rms_stream_test)


print('MAE [Left, Top, Right, Bottom]: {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(overfill_mae_test[0], overfill_mae_test[1], overfill_mae_test[2], overfill_mae_test[3]))
print('RMS: {:.2f}'.format(overfill_rms_test))

#*** NEWEST OPTIMIZATION PART WILL BE HERE***#


#Overfilling Calculation
pred_size = (predictive_values[:,2]-predictive_values[:,0])*(predictive_values[:,1]-predictive_values[:,3])
opt_size = (optimal_values[:,2]-optimal_values[:,0])*(optimal_values[:,1]-optimal_values[:,3])

pred_overfill = (pred_size/4-1)*100
opt_overfill = (opt_size/4-1)*100

overfill99 = np.nanpercentile(pred_overfill,1)

plt.figure()
x = np.sort(pred_overfill)
y = np.arange(1, len(x)+1)/len(x)
plt.plot(x, y, marker='.', linestyle='none')
plt.xlabel('Percentage of OverHead Projection (%)')
plt.ylabel('likehood of Occurance')
plt.title('CDF of Predictive OverHead Anticipation Time: ' + str(anticipation_time))
plt.legend(["99% Percentile : {:.2f}%".format(overfill99)])
plt.margins(0.02)

plt.show()

#plt.figure()
#y = pred_overfill
#y2 = opt_overfill
#y3 = old_overfill
#y4 = np.full(len(pred_Signed),20)
#y5 = np.full(len(pred_Signed),50)
#y6 = np.full(len(pred_Signed),100)
#t = np.arange(0, len(y))/60
#plt.plot(t, y, marker='.', linestyle='none')
#plt.plot(t, y2, marker='.', linestyle='none')
#plt.plot(t, y3, marker='.', linestyle='none')
#plt.plot(t, y4, marker='.', linestyle='none')
#plt.plot(t, y5, marker='.', linestyle='none')
#plt.plot(t, y6, marker='.', linestyle='none')
#plt.xlabel('Percentage of OverHead Projection (%)')
#plt.legend(['Signed','Optimal','Unsigned','fixed 1.2','fixed 1.5', 'fixed 2.0'])
#plt.ylabel('likehood of Occurance')
#plt.title('CDF of OverHead Comparison among Methods')
#plt.margins(0.02)
#
#plt.show()
#
#raw_overfill_data = np.column_stack(
#	[opt_overfill, 
#	old_overfill, 
#	signed2_overfill, 
#	pred_overfill, 
#	y4,
#	y5, 
#	y6]
#    )
#df = pd.DataFrame(raw_overfill_data).to_csv("overfill_data.csv")