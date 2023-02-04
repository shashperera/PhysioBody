# PhysiBody

# Setup
```shell script
pip install -r requirements.txt
```

# Run
```shell script
cd Downloads\physio-pose-master (1)\physio-pose
python physio.py # to view full list
python physio.py --exercise seated_right_knee_extension --joints --skeleton --save-output
python physio.py --exercise left_heel_slides --joints --skeleton --save-output
python physio.py --exercise side_lying_left_leg_lift --joints --skeleton --save-output

python physio.py --video video1.mp4 --csv-path video1.csv #you need 2 videos. Run pose estimation using physio.py on both the videos and save their CSV results
python physio.py --video video1.mp4 --joints --skeleton --save-output
python pose_compare.py video1.csv video2.csv #To calculate how similar the 2 poses run this & output a decimal number. The lower the better.




```

# Exercise Support
| Exercise | Video link | Code | Description |
| --- | --- | --- | --- |
| Heel slides (left) | https://youtu.be/Bz0wSFRjH2c | `left_heel_slides` | Slide the heel towards the buttocks as far as possible. Hold it for 5 seconds and relax. |
| Seated knee flexion and extension (right) | https://youtu.be/OpFov55bKZo | `seated_right_knee_extension` | Best done sitting in a chair. Bend the knee as far as possible and hold for 5sec then straighten as far as possible or bring back to start position. Slowly the range will improve. |
| Side-Lying Leg Lift (left) | https://youtu.be/jgh6sGwtTwk | `side_lying_left_leg_lift` | If you feel unsteady, bend the bottom leg for support. Toes should face forward. |
