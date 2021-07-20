import yan_object_detection

input_image_path = '/Downloads/Applied_pedestrian_crossing_PR.jpeg'

output_detections = yan_object_detection.object_detection_from_image(
    input_image_path)

for d in output_detections:
	print(d)


