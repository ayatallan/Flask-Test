import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load the model from the desktop
model = keras.models.load_model('/home/ayat/Desktop/Kiras/keras_model.h5')

# Compile the model (you can adjust the optimizer, loss, and metrics based on your original compilation)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the class labels from the desktop
with open('/home/ayat/Desktop/Kiras/labels.txt', 'r') as file:
    class_labels = [line.strip() for line in file]

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize the CvBridge
bridge = CvBridge()

# Variable to store the predicted class
predicted_class = ""

def process_and_publish_image(frame):
    global predicted_class

    try:
        # Preprocess the frame for prediction
        frame = cv2.resize(frame, (224, 224))
        img_array = image.img_to_array(frame)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image data

        # Make a prediction using the loaded model
        predictions = model.predict(img_array)

        # Get the class with the highest probability
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_labels[predicted_class_index]

        # Display the frame with the predicted class name
        cv2.putText(frame, f" {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Convert the OpenCV image to a ROS Image message
        ros_image = bridge.cv2_to_imgmsg(frame, encoding="bgr8")

        # Publish the processed image
        # image_pub.publish(ros_image)  # Assuming this line is not needed in Flask

    except Exception as e:
        print(f"Error processing image: {e}")
        predicted_class = "Error processing image"

@app.route('/')
def index():
    return render_template('index.html', predicted_class=predicted_class)

def generate_frames():
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        if not ret:
            break

        # Process the image
        process_and_publish_image(frame)

        # Convert the processed frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    rospy.init_node('image_classifier_node', anonymous=True)

    while not rospy.is_shutdown():
        # Capture a frame from the camera
        ret, frame = cap.read()

        if not ret:
            break

        # Process and publish the image
        process_and_publish_image(frame)

        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run the Flask app in a separate thread
    import threading
    threading.Thread(target=app.run, args=('0.0.0.0', 5000)).start()

    # Run the main function (image processing) in the main thread
    main()
