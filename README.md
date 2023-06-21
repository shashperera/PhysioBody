# PhysiBody

# Setup
```shell script
pip install -r requirements.txt
```

# Run
```shell script
Open anaconda prompt
cd Downloads\physiobody-master\physiobody
python physio.py # to view full list

#Live demos
python physio.py --exercise seated_right_knee_extension --joints --skeleton --save-output
python physio.py --exercise seated_left_knee_extension --joints --skeleton --save-output


python physio.py --exercise left_heel_slides --joints --skeleton --save-output
python physio.py --exercise side_lying_left_leg_lift --joints --skeleton --save-output

#Uploaded videos
    python physio.py --video results.avi --csv-path results.csv #you need 2 videos. Run pose estimation using physio.py on both the videos and save their CSV results
    python physio.py --video SRKE45t.mp4 --joints --skeleton --save-output
    python pose_compare.py check1.csv check2.csv #To calculate how similar the 2 poses run this & output a decimal number. The lower the better.


```

# Flow

Starts by evaluating the movement of the joints. 
1.	First ensure that all of the needed joints have been identified and are visible in the frame. 
2.	The user is then instructed to return to the beginning point.
3.	User is then instructed to complete the remaining steps in the activity. 
4.	The feedback is given based on the angle formed by the keypoints covered.

The application provides guidelines for the user and follows the same directions - Ex: Seated Right Knee Extension

1.	Check whether the keypoints are visible.
2.	Next waits for the user to move the right leg in the seated posture. 
3.	Then waits for the user to extend and straighten the leg.
4.	Next waits for the user to return the leg to the beginning position. 
5.	Also the application gives the user useful output throughout each of the waiting times. For example, if the leg is not in the starting position, directs it to be so.

| Flow | Starts by evaluating the movement of the joints. 
1.	First ensure that all of the needed joints have been identified and are visible in the frame. 
2.	The user is then instructed to return to the beginning point.
3.	User is then instructed to complete the remaining steps in the activity. 
4.	The feedback is given based on the angle formed by the keypoints covered. |
| Application Guideline | The application provides guidelines for the user and follows the same directions - Ex: Seated Right Knee Extension

1.	Check whether the keypoints are visible.
2.	Next waits for the user to move the right leg in the seated posture. 
3.	Then waits for the user to extend and straighten the leg.
4.	Next waits for the user to return the leg to the beginning position. 
5.	Also the application gives the user useful output throughout each of the waiting times. For example, if the leg is not in the starting position, directs it to be so. |





# Exercise Support
| Exercise | Video link | Code | Description |
| --- | --- | --- | --- |
| Heel slides (left) | https://youtu.be/Bz0wSFRjH2c | `left_heel_slides` | Slide the heel towards the buttocks as far as possible. Hold it for 5 seconds and relax. |
| Seated knee flexion and extension (right) | https://youtu.be/OpFov55bKZo | `seated_right_knee_extension` | Best done sitting in a chair. Bend the knee as far as possible and hold for 5sec then straighten as far as possible or bring back to start position. Slowly the range will improve. |
| Side-Lying Leg Lift (left) | https://youtu.be/jgh6sGwtTwk | `side_lying_left_leg_lift` | If you feel unsteady, bend the bottom leg for support. Toes should face forward. |
