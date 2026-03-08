instruction = """
This video is a CCTV-view traffic accident video.
The accident region is a really local area in the whole video so you have to analyze the corners and edges of the video carefully.
Follow the instructions below to analyze the traffic accident video and extract the accident frame (time), accident region, and accident type.

1. Analysis
You should watch the video end to end.
Analyze this video from behinning to end **frame by frame** and gather information about the traffic accident.
Focus mainly on the road and **vehicle movements**. Since the video may include low resolution, occlusion, low-light conditions, and similar challenges, analyze it carefully step by step.
Collision includes both collisions between different vehicles and collisions where a single vehicle hits a stationary object.
**Tracking the movement of vehicles** helps to find the collision moment.

2. Reasoning
Return briefly why did you decide like that.
It might be hard to detect the collision, so you might think there is no collision in the video but there must be a accident(collision) in the video.

3. Temporal Prediction
The video is represented as indexed frames in chronological order.
**Find the indexed frame** where physical contact between vehicles begins or single and the result of the accident after collisions.
Return that two frame index (collision_frame and result_frame).
Then return the corresponding time for that indexed collision_frame.

4. Spatial Prediction
Also return one collision bounding box on that indexed collision_frame using left-top and right-bottom coordinates.
The bbox area include one or two vehicles with collisions directly occurring
The bounding box should enclose the collision region or the involved vehicles at the first contact moment.
If one of the collided vehicles is occluded by other structures, predict the region to include the occluded vehicle as well.
The bounding box must contain at least one vehicle.

5. Type Prediction
Then return the collision type. The collision type includes: head-on, rear-end, sideswipe, single, and t-bone collisions.
Head-on is defined as a collision where the front ends of two vehicles hit each other.
Rear-end is defined as a collision where the front end of one vehicle hits the rear end of another vehicle.
Sideswipe is defined as a slight collision where the sides of two vehicles hit each other.
Single is defined as an accident that involves only one vehicle, such as a vehicle hitting a stationary object or a vehicle losing control and crashing without colliding with another vehicle.
T-bone is defined as a collision where the front end of one vehicle hits the side of another vehicle, forming a 'T' shape.
"""


return_format = """
please return the result in JSON format only, not markdown.
here is the JSON format:
{
    "reasoning ": "explain the situation of the video after accident occurs and why did you decide like that",
    "collision_frame": exact indexed frame where the collision occurs,
    "result_frame" : exact indexed frame after the collision occurs,
    "time": exact time corresponding to that indexed frame,
    "coordinate": [
        [x1, y1],
        [x2, y2]
    ],
    "type": "choose one from [head-on, rear-end, sideswipe, single, t-bone]",
}
---
example:
{
    "reasoning": "After carefully analyzing the video frame by frame, I observed that at frame 150, the front end of a red car made contact with the rear end of a blue car, which indicates a rear-end collision. The bounding box coordinates [100, 200], [300, 400] enclose the area where the two vehicles are in contact. The collision type is classified as rear-end because the red car hit the back of the blue car. After the collision, at frame 180, both vehicles came to a stop, which confirms that the accident occurred.",
    # do not answer like "There is no collision in the video." or "I observed that there is no visible collision"
    "collision_frame": int,
    "result_frame": int,
    "time": "second.millisecond", # without minutes like 00.00
    "coordinate": [
        [x1, y1],
        [x2, y2]
    ],
    "type": "choose one from [head-on, rear-end, sideswipe, single, t-bone]",
}
"""