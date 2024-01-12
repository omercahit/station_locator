#!/usr/bin/env python3
import rospy
import tf
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped
from actionlib import SimpleActionClient

class MoveBaseWithTF:
    def __init__(self):
        rospy.init_node('move_base_with_tf', anonymous=True)
        
        # tf listener oluştur
        self.listener = tf.TransformListener()

        # MoveBaseClient oluştur
        self.move_base = SimpleActionClient('move_base', MoveBaseAction)
        self.move_base.wait_for_server()

    def get_target_pose(self, target_frame, source_frame='qr_frame'):
        try:
            # İki çerçeve arasındaki dönüşümü al
            self.listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
            trans, rot = self.listener.lookupTransform(target_frame, source_frame, rospy.Time(0))

            print(trans)

            # Dönüşümü PoseStamped formatına dönüştür
            target_pose = PoseStamped()
            target_pose.header.frame_id = target_frame
            target_pose.pose.position.x = trans[0] - 1.2
            target_pose.pose.position.y = trans[1]
            target_pose.pose.position.z = trans[2]
            target_pose.pose.orientation.x = rot[0]
            target_pose.pose.orientation.y = rot[1]
            target_pose.pose.orientation.z = rot[2]
            target_pose.pose.orientation.w = rot[3]

            return target_pose

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Dönüşüm alınamadı!")

    def move_to_target(self, target_frame):
        # Hedef pozisyonu al
        target_pose = self.get_target_pose(target_frame)
        print(target_pose)

        # MoveBaseGoal oluştur
        goal = MoveBaseGoal()
        goal.target_pose = target_pose

        # MoveBase'a hedefi gönder
        self.move_base.send_goal(goal)

        # Hareketin tamamlanmasını bekleyin
        self.move_base.wait_for_result()

if __name__ == '__main__':
    try:
        move_base_with_tf = MoveBaseWithTF()

        # Hedef çerçeve adı
        target_frame = 'map'

        # Belirtilen hedefe hareket et
        move_base_with_tf.move_to_target(target_frame)

    except rospy.ROSInterruptException:
        pass
