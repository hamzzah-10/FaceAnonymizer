import cv2
import mediapipe as mp
import argparse
import os

def process_img(img, face_detection, H, W):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
    
            # Convert relative coordinates to absolute pixel values
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Increase the size of the bounding box by 20%
            scale_factor = 0.2
            x1 = int(x1 - w * scale_factor / 2)
            y1 = int(y1 - h * scale_factor)
            w = int(w * (1 + scale_factor))
            h = int(h * (1 + scale_factor))
            
            # Ensure the bounding box doesn't go out of the image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            w = min(W - x1, w)
            h = min(H - y1, h)

            # Blur the face
            img[y1:y1+h, x1:x1+w] = cv2.blur(img[y1:y1+h, x1:x1+w], (100, 100))

    return img

def process_video(cap, face_detection, output_path=None):
    # Set up video writer if an output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame = process_img(frame, face_detection, H, W)

        cv2.imshow("Video", frame)

        if output_path:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def main(args):
    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Detect faces
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        if args.mode == 'image':
            img = cv2.imread(args.filePath)
            H, W, _ = img.shape
            img = process_img(img, face_detection, H, W)

            # Save image
            output_path = os.path.join(args.output_dir, "output.jpg")
            cv2.imwrite(output_path, img)
            print(f"Processed image saved to {output_path}")

        elif args.mode == 'video':
            if args.filePath == 'webcam':
                cap = cv2.VideoCapture(0)  # Open webcam
                output_path = os.path.join(args.output_dir, "output.avi")
            else:
                cap = cv2.VideoCapture(args.filePath)
                output_path = os.path.join(args.output_dir, "output.avi")

            process_video(cap, face_detection, output_path)
            print(f"Processed video saved to {output_path}")

        elif args.mode == 'webcam':
            cap = cv2.VideoCapture(0)  # Open webcam
            process_video(cap, face_detection)
            print("Webcam feed processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default='image', help="Mode of operation: image, video, or webcam")
    parser.add_argument("--filePath", default="img.jpeg", help="Path to the image or video file or 'webcam' for webcam mode")
    parser.add_argument("--output_dir", default="F:\\OpenCV\\Project 2 - Face Anonymizer\\output", help="Directory to save the output")

    args = parser.parse_args()

    main(args)
