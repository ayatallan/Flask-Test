# app.py
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, Response, send_file
import io
import matplotlib.pyplot as plt
import base64

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

def read_pgm(file_path):
    with open(file_path, 'rb') as f:
        # Read the magic number (P5)
        magic_number = f.readline().decode().strip()
        if magic_number != 'P5':
            raise ValueError("Invalid PGM file format")

        # Skip comment lines
        while True:
            line = f.readline().decode().strip()
            if not line.startswith('#'):
                break

        # Read width, height, and maximum pixel value
        width, height = map(int, line.split())
        max_pixel_value = int(f.readline().decode().strip())

        # Read pixel data
        pixel_data = bytearray(f.read())

    return width, height, max_pixel_value, pixel_data

def generate_pgm_data_uri(file_path):
    width, height, max_pixel_value, pixel_data = read_pgm(file_path)
    image_array = np.frombuffer(pixel_data, dtype=np.uint8).reshape((height, width))

    # Create a larger Matplotlib figure
    plt.figure(figsize=(10, 10))

    # Use a grayscale colormap
    plt.imshow(image_array, cmap='gray', vmin=0, vmax=max_pixel_value)

    # Remove the color bar
    # plt.colorbar()

    plt.title('SLAM 2D Map (Grayscale)')
    plt.axis('off')  # Turn off axis labels

    # Save the Matplotlib figure to a BytesIO object
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    # Close the Matplotlib figure to free up resources
    plt.close()

    # Convert the image buffer to a data URI
    pgm_data_uri = "data:image/png;base64," + base64.b64encode(img_buffer.read()).decode('utf-8')

    return pgm_data_uri
@app.route('/download_result_pdf')
def download_result_pdf():
    # PGM map data URI
    pgm_map_data_uri = generate_pgm_data_uri("/home/ayat/Desktop/flask/flask_pgm_viewer/gmapping_01.pgm")

    # Render the result.html template with the prediction and PGM map data URI
    html_content = render_template('result.html', predicted_class=predicted_class, pgm_map_data_uri=pgm_map_data_uri)

    # Create a PDF using html2pdf.js
    pdf_file = html2pdf.create_pdf(html_content)

    # Create a response with the PDF file
    response = make_response(pdf_file)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'inline; filename=result.pdf'

    return response
@app.route('/result')
def result():
    # PGM map data URI
    pgm_map_data_uri = generate_pgm_data_uri("/home/ayat/Desktop/flask/flask_pgm_viewer/gmapping_01.pgm")

    # Render the result.html template with the prediction and PGM map data URI
    return render_template('result.html', predicted_class=predicted_class, pgm_map_data_uri=pgm_map_data_uri)

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
