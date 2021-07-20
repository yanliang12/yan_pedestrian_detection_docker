# yan_pedestrian_detection_docker

```bash 
docker pull yanliang12/yan_pedestrian_detection:1.0.1
```

```python
import yan_object_detection

input_image_path = 'Applied_pedestrian_crossing_PR.jpeg'

output_detections = yan_object_detection.object_detection_from_image(
    input_image_path)

for d in output_detections:
	print(d)

'''
{'x1': 237, 'x2': 290, 'y1': 803, 'y2': 1041, 'label': 'person', 'score': 0.9997958540916443}
'''
```
