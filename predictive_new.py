# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:16:39 2020

@author: isfan fauzi
"""

import numpy as np
import csv as csv
import pandas as pd
import matplotlib.pyplot as plt
import quaternion
from rms import rms
import math
import time

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

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
	
#	return [-np.abs(p_l),np.abs(p_t),np.abs(p_r),-np.abs(p_b)]
	margins = np.max(np.abs([p_l, p_t, p_r, p_b]))
	return [-margins, margins, margins, -margins]

def Robust_overfilling(input_orientation, prediction, input_projection, fixed_param):
	IPD = input_projection[2]-input_projection[0]
	h = 1
	r = np.sqrt(h**2+1/4*IPD**2)
	
	input_angle = np.arctan(IPD/(2*h))
	pitch_diff = prediction[0]-input_orientation[0]
	roll_diff = prediction[1]-input_orientation[1]
	yaw_diff = prediction[2]-input_orientation[2]
	
	x_r = max(input_projection[2],r*np.sin(input_angle-yaw_diff))
	x_l = min(input_projection[0],-r*np.sin(input_angle+yaw_diff))
	
	y_t = max(input_projection[1],r*np.sin(input_angle+pitch_diff))
	y_b = min(input_projection[3],-r*np.sin(input_angle-pitch_diff))

	if (roll_diff<0):
		x_rr = r*np.sin(input_angle-roll_diff)
		x_ll = -r*np.sin(input_angle-roll_diff)
		y_tt = r*np.cos(input_angle+roll_diff)
		y_bb = -r*np.cos(input_angle+roll_diff)
	else:
		x_rr = r*np.sin(input_angle+roll_diff)
		x_ll = -r*np.sin(input_angle+roll_diff)
		y_tt = r*np.cos(input_angle-roll_diff)
		y_bb = -r*np.cos(input_angle-roll_diff)
	
	p_r = x_r+x_rr-IPD/2
	p_l = x_l+x_ll+IPD/2
	p_t = y_t+y_tt-IPD/2
	p_b = y_b+y_bb+IPD/2
	
#	p_r = np.sqrt(abs(p_r-p_l)**2+1)-abs(p_l)
#	p_l = np.sqrt(abs(p_l-p_r)**2+1)-p_r
#	p_t = np.sqrt(abs(p_t-p_b)**2+1)-abs(p_b)
#	p_b = np.sqrt(abs(p_b-p_t)**2+1)-p_t
	
	p_r = max(p_r*(np.sin(abs(yaw_diff))*(fixed_param-1)+1),p_r*(np.sin(abs(pitch_diff))*(fixed_param-1)+1))
	p_l = min(p_l*(np.sin(abs(yaw_diff))*(fixed_param-1)+1),p_l*(np.sin(abs(pitch_diff))*(fixed_param-1)+1))
	p_t = max(p_t*(np.sin(abs(pitch_diff))*(fixed_param-1)+1),p_t*(np.sin(abs(yaw_diff))*(fixed_param-1)+1))
	p_b = min(p_b*(np.sin(abs(pitch_diff))*(fixed_param-1)+1), p_b*(np.sin(abs(yaw_diff))*(fixed_param-1)+1))
	
#	return [-np.abs(p_l),np.abs(p_t),np.abs(p_r),-np.abs(p_b)]
	margins = np.max(np.abs([p_l, p_t, p_r, p_b]))
	return [-margins, margins, margins, -margins]

# File name of the trace file
inFile = "s20200915_scene(3)_user(1)_(300).csv"
outFile1 = "200915_scene(3)_user(1)_(20)_optimal-omni.csv"
outFile2 = "20200915_scene(3)_user(1)_(300)_robust_predictive.csv"

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
MAE_error = np.nanmean(np.abs(signed_pred_error),axis=0)
print("MAE Error: Pitch, Roll, Yaw:")
print(MAE_error)

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

opt_time = time.time() - st
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

## Calculate initial of predicted values
input_orientation = orie_data[:-anticipation_size]
prediction = pred_orie_data[anticipation_size:]	# current quaternion orientation
input_projection = proj_data		# predicted quaternion orientation, obtained (anticipation_time) before

robust_values = np.empty((0,4), dtype=np.float32)

st = time.time()
for i in range(0, len(input_orientation)+anticipation_size):
	if (i<len(input_orientation)):
		overhead_val = Robust_overfilling(input_orientation[i], prediction[i], input_projection[i], 1.5)
		robust_values = np.vstack((robust_values,overhead_val))
	else:
		robust_values = np.vstack((robust_values, input_projection[i]))

robust_time = time.time() - st
raw_robust_data = np.column_stack(
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
	robust_values
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

# write the predicted into file 
df = pd.DataFrame(raw_robust_data,columns = ['timestamp',
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

## Check
#print(optimal_values)
#print(pred_values)

# plot margins just to check
plt.figure()
plt.plot((timestamp-timestamp[0])/705600000, abs(optimal_values[:, 0])-1, linewidth=1)
plt.plot((timestamp-timestamp[0])/705600000, abs(robust_values[:, 0])-1, linewidth=1)
plt.legend(['Optimal','Robust'])
plt.title('Margins (Left) Comparison')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Margins')

# plot margins just to check
plt.figure()
plt.plot((timestamp-timestamp[0])/705600000, abs(optimal_values[:, 1])-1, linewidth=1)
plt.plot((timestamp-timestamp[0])/705600000, abs(robust_values[:, 1])-1, linewidth=1)
plt.legend(['Optimal','Robust'])
plt.title('Margins (Top) Comparison')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Margins')

# plot margins just to check
plt.figure()
plt.plot((timestamp-timestamp[0])/705600000, abs(optimal_values[:, 2])-1, linewidth=1)
plt.plot((timestamp-timestamp[0])/705600000, abs(robust_values[:, 2])-1, linewidth=1)
plt.legend(['Optimal','Robust'])
plt.title('Margins (Right) Comparison')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Margins')

# plot margins just to check
plt.figure()
plt.plot((timestamp-timestamp[0])/705600000, abs(optimal_values[:, 3])-1, linewidth=1)
plt.plot((timestamp-timestamp[0])/705600000, abs(robust_values[:, 3])-1, linewidth=1)
plt.legend(['Optimal','Robust'])
plt.title('Margins (Bottom) Comparison')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Margins')

plt.show()

#Overfilling Calculation
robust_size = (robust_values[:,2]-robust_values[:,0])*(robust_values[:,1]-robust_values[:,3])
opt_size = (optimal_values[:,2]-optimal_values[:,0])*(optimal_values[:,1]-optimal_values[:,3])

robust_overfill = (robust_size/4-1)*100
opt_overfill = (opt_size/4-1)*100

overfill99 = np.nanpercentile(robust_overfill,99)

plt.figure()
x = np.sort(robust_overfill)
y = np.arange(1, len(x)+1)/len(x)
plt.plot(x, y, marker='.', linestyle='none')
plt.xlabel('Percentage of OverHead Projection (%)')
plt.ylabel('likehood of Occurance')
plt.title('CDF of Robust-Omni OverHead Anticipation Time: ' + str(anticipation_time))
plt.legend(["99% Percentile : {:.2f}%".format(overfill99)])
plt.margins(0.02)

plt.show()
