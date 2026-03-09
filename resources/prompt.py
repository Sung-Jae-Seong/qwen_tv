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


2. Temporal Prediction
The video is represented as indexed frames in chronological order.
**Find the indexed frame** where physical contact between vehicles begins or single(collision_frame).

3. Spatial Prediction
Also return one collision bounding box on that indexed collision_frame using left-top and right-bottom coordinates.
The bbox area include one or two vehicles with collisions directly occurring
The bounding box should enclose the collision region or the involved vehicles at the first contact moment.
If one of the collided vehicles is occluded by other structures, predict the region to include the occluded vehicle as well.
The bounding box must contain at least one vehicle.

4. Type Prediction
Then return the collision type. The collision type includes: head-on, rear-end, sideswipe, single, and t-bone collisions.
Head-on is defined as a collision where the front ends of two vehicles hit each other.
Rear-end is defined as a collision where the front end of one vehicle hits the rear end of another vehicle.
Sideswipe is defined as a slight collision where the sides of two vehicles hit each other.
Single is defined as an accident that involves only one vehicle, such as a vehicle hitting a stationary object or a vehicle losing control and crashing without colliding with another vehicle.
T-bone is defined as a collision where the front end of one vehicle hits the side of another vehicle, forming a 'T' shape.

5. Reasoning
Return briefly why did you decide like that.
It might be hard to detect the collision, so you might think there is no collision in the video but there must be a accident(collision) in the video.
"""


return_format = """
please return the result in JSON format only, not markdown.
here is the JSON format:
{
    "collision_frame": exact indexed frame where the collision occurs,
    "coordinate": [
        [x1, y1],
        [x2, y2]
    ],
    "type": "choose one from [head-on, rear-end, sideswipe, single, t-bone]",
    "reasoning ": "explain the situation of the video after accident occurs and why did you decide like that",
}
---
example:
{
    "collision_frame": int,
    "coordinate": [
        [x1, y1],
        [x2, y2]
    ],
    "type": "choose one from [head-on, rear-end, sideswipe, single, t-bone]",
    "reasoning": "I observed ... at frame ... , ... made contact , which indicates ... The bounding box coordinates ... enclose the area where ...
}
"""