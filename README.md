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
