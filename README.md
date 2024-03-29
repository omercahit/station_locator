# Station Locator

Clone this repository into your workspace.

```bash
cd (YOUR_WORKSPACE_FOLDER)/src
git clone https://github.com/omercahit/station_locator.git
```
Run your realsense node and check the topic names of rgb and depth images with this command:
```bash
rostopic list | grep camera
```
If there is any difference between yours and this package, you can edit the names in the image_subscriber() function which is in the station_locator.py.

If you want to change the qr code with another qr code or any image you want, go to "resources" folder and replace your image with "qr_code_big.png".

To run the ROS node, go to directory of package and get into the src folder.

```bash
cd (YOUR_WORKSPACE_FOLDER)/src/station_locator/src
```

If the realsense node is up, run the station_locator node as following:

```bash
rosrun station_locator station_locator.py
```

After running the station_locator node, qr_frame will be published as a tf.

## Gazebo Simulation Scene:

![Screenshot from 2024-01-29 09-56-39](https://github.com/omercahit/station_locator/assets/35147835/3ac72ed0-4a63-4054-87a9-08b093ff2dab)

## View of qr_frame tf on Rviz:

![Screenshot from 2024-01-29 09-59-59](https://github.com/omercahit/station_locator/assets/35147835/76ccee29-91aa-427f-ab06-ab9e3ca51d65)

## Send Move Base Goal to The Frame
If you want to send move_base command to your robot to go to the located frame, you can use this command:

```bash
rosrun station_locator move_base_to_qr_frame.py
```
This will send your robot to a position which will be 1.2m between the base_link of the robot and the qr_frame.
