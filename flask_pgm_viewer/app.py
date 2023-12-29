from flask import Flask, render_template, send_file
import io
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

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

def generate_pgm_image(file_path):
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

    return img_buffer

@app.route('/view_pgm')
def view_pgm():
    # Replace with the actual path to your SLAM-generated 2D map
    pgm_file_path = '/home/ayat/Desktop/flask/flask_pgm_viewer/gmapping_01.pgm'
    img_buffer = generate_pgm_image(pgm_file_path)
    return send_file(img_buffer, mimetype='image/png')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
