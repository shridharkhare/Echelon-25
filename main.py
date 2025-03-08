from ultralytics import YOLO
import os
import cv2
import numpy as np
from pathlib import Path

# Step 1: Tile images and adjust labels
def tile_images_and_labels(image_dir, label_dir, output_image_dir, output_label_dir, tile_size=640, overlap=0.1):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for img_file in os.listdir(image_dir):
        if not img_file.endswith(".jpg"):
            continue

        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, f"{Path(img_file).stem}.txt")
        if not os.path.exists(label_path):
            print(f"Label file not found for {img_file}, skipping...")
            continue

        # Load image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Load labels
        with open(label_path, "r") as f:
            labels = [line.strip().split() for line in f.readlines()]

        # Define tiling parameters
        step_size = int(tile_size * (1 - overlap))
        tile_idx = 0

        for y in range(0, h, step_size):
            for x in range(0, w, step_size):
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                tile = img[y:y_end, x:x_end]
                if tile.size == 0:
                    continue

                # Pad tile if smaller than tile_size
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    pad_h = max(0, tile_size - tile.shape[0])
                    pad_w = max(0, tile_size - tile.shape[1])
                    tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

                # Save tile
                tile_filename = f"{Path(img_file).stem}tile{tile_idx}.jpg"
                tile_path = os.path.join(output_image_dir, tile_filename)
                cv2.imwrite(tile_path, tile)

                # Adjust labels for this tile
                tile_labels = []
                tile_h, tile_w = tile.shape[:2]
                for label in labels:
                    class_id, x_center, y_center, width, height = map(float, label)

                    # Convert normalized coordinates to absolute
                    x_center_abs = x_center * w
                    y_center_abs = y_center * h
                    width_abs = width * w
                    height_abs = height * h

                    # Check if the bounding box intersects with the tile
                    x1 = x_center_abs - (width_abs / 2)
                    y1 = y_center_abs - (height_abs / 2)
                    x2 = x_center_abs + (width_abs / 2)
                    y2 = y_center_abs + (height_abs / 2)

                    # Check if box is within tile bounds
                    if x2 < x or x1 > x_end or y2 < y or y1 > y_end:
                        continue

                    # Adjust coordinates relative to the tile
                    x_center_tile = (x_center_abs - x) / tile_w
                    y_center_tile = (y_center_abs - y) / tile_h
                    width_tile = width_abs / tile_w
                    height_tile = height_abs / tile_h

                    # Ensure the bounding box is within the tile
                    if x_center_tile < 0 or x_center_tile > 1 or y_center_tile < 0 or y_center_tile > 1:
                        continue

                    tile_labels.append(f"{int(class_id)} {x_center_tile:.6f} {y_center_tile:.6f} {width_tile:.6f} {height_tile:.6f}")

                # Save adjusted labels
                tile_label_path = os.path.join(output_label_dir, f"{Path(img_file).stem}tile{tile_idx}.txt")
                with open(tile_label_path, "w") as f:
                    f.write("\n".join(tile_labels))

                tile_idx += 1

        print(f"Tiled {img_file} into {tile_idx} tiles with adjusted labels.")

# Step 2: Train the YOLOv8 model on tiled images with manual validation per epoch
def train_yolov8():
    # Load a pre-trained model
    model = YOLO("yolov8m.pt")

    # Training parameters
    project_dir = Path(_file_).parent.absolute()
    data_yaml = project_dir / "dataset" / "data_tiled.yaml"
    epochs = 25
    imgsz = 640
    batch = 2
    device = "cpu"
    project = "runs/train"
    name = "exp"
    augment = True
    mosaic = 1.0
    mixup = 0.2
    patience = 10

    # Custom training loop to print metrics per epoch
    for epoch in range(epochs):
        print(f"\nStarting Epoch {epoch}/{epochs - 1}...")
        # Train for one epoch
        model.train(
            data=str(data_yaml),
            epochs=1,  # Train for 1 epoch at a time
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name if epoch == 0 else f"{name}epoch{epoch}",
            augment=augment,
            mosaic=mosaic,
            mixup=mixup,
            patience=patience,
            resume=epoch > 0,  # Resume training from the last epoch
            verbose=False
        )

        # Run validation to get metrics
        print(f"Validating Epoch {epoch}/{epochs - 1}...")
        metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device, verbose=False)
        print(f"Epoch {epoch}/{epochs - 1}: "
              f"Precision: {metrics.box.map:.4f}, "
              f"Recall: {metrics.box.map50:.4f}, "
              f"mAP@0.5: {metrics.box.map50:.4f}, "
              f"mAP@0.5:0.95: {metrics.box.map75:.4f}")

    print("Training completed! Weights saved in runs/train/exp/weights/")

# Step 3: Test the model on original images
def test_yolov8():
    # Load the trained model
    project_dir = Path(_file_).parent.absolute()
    model = YOLO(project_dir / "runs/train/exp8/weights/best.pt")

    # Define test images (original images)
    test_images = [
        project_dir / "dataset/images/train/4.jpg",
        project_dir / "dataset/images/val/5.jpg"
    ]

    output_dir = project_dir / "output"
    os.makedirs(output_dir, exist_ok=True)

    for test_image_path in test_images:
        if not test_image_path.exists():
            print(f"Image not found: {test_image_path}")
            continue

        # Load the original image
        orig_img = cv2.imread(str(test_image_path))
        h, w = orig_img.shape[:2]

        # Define tile size and overlap
        tile_size = 640
        overlap = 0.1
        step_size = int(tile_size * (1 - overlap))

        # Lists to store all detections
        all_boxes = []
        all_scores = []
        all_class_ids = []

        # Divide image into tiles and predict
        for y in range(0, h, step_size):
            for x in range(0, w, step_size):
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                tile = orig_img[y:y_end, x:x_end]
                if tile.size == 0:
                    continue

                # Pad tile if smaller than tile_size
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    pad_h = max(0, tile_size - tile.shape[0])
                    pad_w = max(0, tile_size - tile.shape[1])
                    tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

                # Predict on tile
                results = model.predict(
                    source=cv2.cvtColor(tile, cv2.COLOR_BGR2RGB),
                    conf=0.05,
                    imgsz=tile_size
                )

                # Process results
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    for box, score, class_id in zip(boxes, scores, class_ids):
                        x1, y1, x2, y2 = box
                        x1 += x
                        x2 += x
                        y1 += y
                        y2 += y
                        all_boxes.append([x1, y1, x2, y2])
                        all_scores.append(score)
                        all_class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS)
        if all_boxes:
            boxes = np.array(all_boxes)
            scores = np.array(all_scores)
            class_ids = np.array(all_class_ids)

            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                score_threshold=0.05,
                nms_threshold=0.5
            )
            if len(indices) > 0:
                indices = indices.flatten()
                all_boxes = boxes[indices].tolist()
                all_scores = scores[indices].tolist()
                all_class_ids = class_ids[indices].tolist()
            else:
                all_boxes = []
                all_scores = []
                all_class_ids = []

        # Draw bounding boxes on original image
        output_img = orig_img.copy()
        class_names = {i: f"symbol_{i}" for i in range(32)}
        for (x1, y1, x2, y2), score, class_id in zip(all_boxes, all_scores, all_class_ids):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_names[class_id]}: {score:.2f}"
            cv2.putText(output_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the output image
        output_filename = f"{test_image_path.stem}_with_boxes.jpg"
        output_path = output_dir / "detected_symbols" / output_filename
        os.makedirs(output_dir / "detected_symbols", exist_ok=True)
        cv2.imwrite(str(output_path), output_img)
        print(f"Results for {test_image_path.name} saved to {output_path}")
        if all_boxes:
            print(f"Detected {len(all_boxes)} symbols in {test_image_path.name}.")
        else:
            print(f"No symbols detected in {test_image_path.name}.")

# Step 4: Evaluate model accuracy
def evaluate_accuracy():
    project_dir = Path(_file_).parent.absolute()
    model = YOLO(project_dir / "runs/train/exp/weights/best.pt")
    print("Evaluating final model accuracy...")
    metrics = model.val()
    print(f"Final Validation Results:")
    print(f"Precision: {metrics.box.map:.4f}")
    print(f"Recall: {metrics.box.map50:.4f}")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map75:.4f}")
    print(f"Results saved in runs/train/exp/")

# Main execution
if _name_ == "_main_":
    try:
        # Tile the training and validation images
        project_dir = Path(_file_).parent.absolute()
        tile_images_and_labels(
            str(project_dir / "dataset/images/train"),
            str(project_dir / "dataset/labels/train"),
            str(project_dir / "dataset/images/train_tiled"),
            str(project_dir / "dataset/labels/train_tiled")
        )
        tile_images_and_labels(
            str(project_dir / "dataset/images/val"),
            str(project_dir / "dataset/labels/val"),
            str(project_dir / "dataset/images/val_tiled"),
            str(project_dir / "dataset/labels/val_tiled")
        )

        # Create a new data.yaml for the tiled dataset
        tiled_data_yaml = project_dir / "dataset" / "data_tiled.yaml"
        with open(tiled_data_yaml, "w") as f:
            f.write(
                f"train: {project_dir / 'dataset/images/train_tiled'}\n"
                f"val: {project_dir / 'dataset/images/val_tiled'}\n"
                f"nc: 32\n"
                f"names: ['symbol_0', 'symbol_1', 'symbol_2', 'symbol_3', 'symbol_4', 'symbol_5', 'symbol_6', 'symbol_7', 'symbol_8', 'symbol_9', 'symbol_10', 'symbol_11', 'symbol_12', 'symbol_13', 'symbol_14', 'symbol_15', 'symbol_16', 'symbol_17', 'symbol_18', 'symbol_19', 'symbol_20', 'symbol_21', 'symbol_22', 'symbol_23', 'symbol_24', 'symbol_25', 'symbol_26', 'symbol_27', 'symbol_28', 'symbol_29', 'symbol_30', 'symbol_31']\n"
            )

        # Train the model on tiled images
        train_yolov8()

        # Test the model on original images
        test_yolov8()

        # Evaluate accuracy
        evaluate_accuracy()

    except Exception as e:
        print(f"An error occurred: {e}")