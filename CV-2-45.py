import cv2
import numpy as np
from skimage.data import astronaut
from jsonargparse import CLI

def process_image(image: np.ndarray, delta: float) -> np.ndarray:
    """Changes S (saturation) channel in HSV image"""
    # Convert to float32 for safe multiplication
    image_cpy = image.astype(np.float32)
    image_cpy[:, :, 1] = image_cpy[:, :, 1] * delta

    # Clip and convert back to uint8
    image_cpy = np.clip(image_cpy, 0, 255).astype(np.uint8)

    return image_cpy

def on_trackbar(val):
    """Callback for trackbar to update saturation"""
    global image_hsv, image_new
    delta = val / 100.0  # Convert trackbar value (0-200) to delta (0.0-2.0)
    image_hsv_modified = process_image(image_hsv, delta)
    image_new = cv2.cvtColor(image_hsv_modified, cv2.COLOR_HSV2BGR)
    cv2.imshow("Adjusted Image", image_new)

def main(image_path: str | None = None, output_path: str = "output_image.jpg"):
    """Main function with trackbar for saturation adjustment and save option"""
    global image_hsv, image_new

    # Load image
    if image_path is None:
        image = astronaut()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

    assert len(image.shape) == 3 and image.shape[-1] == 3, "Image must be a 3-channel color image"

    # Convert to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_new = image.copy()  # Initialize image_new

    # Create window and trackbar
    cv2.namedWindow("Adjusted Image")
    cv2.createTrackbar("Saturation", "Adjusted Image", 100, 200, on_trackbar)

    print("Use trackbar to adjust saturation. Press 's' to save, 'q' to quit.")

    while True:
        cv2.imshow("Original", image)
        cv2.imshow("Adjusted Image", image_new)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(output_path, image_new)
            print(f"Image saved to {output_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    CLI(main, as_positional=False)