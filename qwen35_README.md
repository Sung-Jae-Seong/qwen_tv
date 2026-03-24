# 결론: 2-stage로 classification 진행해보자
1. 9B에게 new prompt 버전 기준으로 single vs multi 2진 분류하는 프롬프트로 수정해서 진행하기
    - 프롬프트 핵심은 In-transport와 Not In-transport 정의가 있다는 것
    - 각 사고 유형별 설명은 없애고 오로지 single 설명과 single이 아닌 설명만 있도록
        - a. new prompt에서 task만 2진 분류하는 것으로 바꿔서 수정해보기
        - b. Perception - Cognition - Answer 과정을 거치도록 프롬프트 수정해보기
2. multi 내에서 4개 유형으로 분류하도록 시키기. 이 때 프롬프트는 legacy 버전.
    - 시스템 프롬프트로 이 비디오는 무조건 multi 사고라는 것을 명시하기
    - single 유형에 대한 설명 없애기
    - 각 유형별 설명하는 부분은 new prompt에 명시되어 있는 것을 참고해서 수정해도 괜찮을 듯
        - a. 기존 프롬프트에서 task만 4개 유형으로 분류하는 것으로 바꿔서 수정해보기
        - b. Perception - Cognition - Answer 과정을 거치도록 프롬프트 수정해보기
3. 1번과 2번 결과를 하나로 합쳐서 리안님 csv(score 0.43xxx)에 type 값만 replace 해서 저장하기
4. 디버깅 사이트에 업로드해서 type 정확도 확인하기 & error 심층 분석에 들어가서 type 유형 별 error 확인하기
5. 1~4번 과정을 qwen3-8b-instruct에 대해서도 진행해보기
6. 목표는 type 유형 error를 최소화 하기
    - 이를 위해 프롬프트 수정하거나 9b 8b 모델 비교해보거나
    - 우선 9b를 중점적으로 진행
------------------------------------------------------
new 프롬프트 by ANSI D16 & FMCA-2022

```python
System Prompt:
You are an AI vision expert tasked with analyzing CCTV traffic accident footage and classifying the crash type. You must strictly apply the following statistical and crash classification standards based on ANSI D16.1 and FMCSA-2022.

[CRITICAL RULE: Focus on the Actual Impact]
You must evaluate the crash based strictly on the exact moment of the actual collision (the "First Harmful Event"). Do not classify the crash based on precursors, near-misses, sudden braking, or evasive swerving leading up to the event. The classification must strictly reflect the physical impact itself.

[Core Concept: Vehicle Status (ANSI D16.1)]

In-Transport: A vehicle that is in motion or stopped within the portion of the roadway ordinarily used by similar vehicles. This includes disabled vehicles abandoned in an active travel lane.

Not In-Transport: Any transport vehicle that is not "in-transport." This includes vehicles legally parked off the roadway, or motionless vehicles completely stopped on the shoulder, median, or roadside.

[Crash Classification Criteria (FMCSA-2022 & ANSI combined)]
You must classify the crash in the provided video into exactly ONE of the following 5 categories based on the actual impact:

Single (Single-Vehicle Crash): A crash involving only ONE "In-Transport" vehicle. CRITICAL RULE: If an "In-Transport" vehicle collides with a "Not In-Transport" vehicle (e.g., a legally parked car, or a disabled vehicle completely stopped on the shoulder), it is NOT a collision between two in-transport vehicles. Therefore, it MUST be classified as a Single crash.

Head-on: A crash where two "In-Transport" vehicles traveling in opposite directions collide front-to-front.

Rear-end: A crash where two "In-Transport" vehicles are traveling in the same direction, and the front of the trailing vehicle strikes the rear of the leading vehicle (Front-to-Rear).

Sideswipe: A crash where two "In-Transport" vehicles (traveling in the same or opposite directions) collide side-to-side in a glancing or sliding manner.

T-bone / Angle: A crash where the front of one "In-Transport" vehicle strikes the side of another "In-Transport" vehicle at or near a right angle.

[Instruction]
Analyze the provided video carefully. Fast-forward to the exact moment of impact. Identify the number of "In-Transport" vehicles involved, observe the vehicle trajectories and the initial points of impact, and output exactly one crash type from the 5 categories above that best describes the event.
```

legacy 프롬프트
```python
instruction = (
        "return the time when the collision occurs, "
        "and return the collision bounding box with left-top and right-bottom coordinates. "
        "And return the collision type. The collision type includes: head-on, rear-end, sideswipe, single, "
        "and t-bone collisions. "
        "Head-on is defined as a collision where the front ends of two vehicles hit each other. "
        "Rear-end is defined as a collision where the front end of one vehicle hits the rear end of another vehicle. "
        "Sideswipe is defined as a slight collision where the sides of two vehicles hit each other. "
        "Single is defined as an accident that involves only one vehicle, such as a vehicle hitting a stationary object or a vehicle losing control and crashing without colliding with another vehicle. "
        "T-bone is defined as a collision where the front end of one vehicle hits the side of another vehicle, forming a 'T' shape."
    )

    return_format = """
please return the result in JSON format only, not markdown.
here is the JSON format:
{
    "time": exact time, the temporal location of the video where collision occured,
    "coordinate": left-top and right-bottom, the position of bounding box on the video frame that contains the collision,
    "type": choose and return one of the following [head-on, rear-end, sideswipe, single, t-bone],
    "why": explain why did you return that time, coordinate and type.
}
---
example:
{
    "time": "second.milisecond", # do not return time in hh:mm:ss format, for example, if the collision occurs at 1 second and 500 milliseconds, please return 1.5
    "coordinate": [
        [x1, y1],
        [x2, y2]
    ],
    "type": "one of the following [head-on, rear-end, sideswipe, single, t-bone]",
    "why": "..."
}
"""
```
--------------------------------------------
### Perception - Cognition - Answer 과정 참고

paper: Vad-R1: Towards Video Anomaly Reasoning via Perception-to-Cognition Chain-of-Thought

3.1 Perception-to-Cognition Chain-of-Thought

코드도 공개되어 있으니 해당 프롬프트 부분 참고해서 수정하면 좋을듯

---------------------------------------------
* 모델 이미 다운 받은게 있어서 그 경로를 기준으로 진행함
* 새로 할 때는 모델명만 기입해서 다운 받아야 함.
* 새로 다운로드해야 하는 경우:
  - --model_name_or_path Qwen/Qwen3.5-9B 사용
  - --local_files_only true 는 제거하거나 false 로 둠

9b 추론 돌리기
```bash
CUDA_VISIBLE_DEVICES=0,1 nohup python3 inference_qwen35_9b_base_video_classification.py \
  --model_name_or_path /home/milab/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local_files_only true \
  --base_submission_csv submission_rian.csv \
  --max_new_tokens 384 \
  --experiment_name qwen35_9b_base_full_video_classification_gpu67 \
  > infer_qwen35_9b_base_full_video_classification_gpu67.log 2>&1 &
```
------------------------------------------------
27b-fp8로 진행

* vllm 서버 띄우기
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve \
  /home/milab/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B-FP8/snapshots/2e1b21350ce589fcaafbb3c7d7eac526a7aed582 \
  --tensor-parallel-size 2 \
  --host 127.0.0.1 \
  --port 8000 \
  --enforce-eager \
  --disable-custom-all-reduce \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  > vllm_27b.log 2>&1 &
```

* 비디오 서버 띄우기
```bash
nohup python3 -m http.server 8002 \
  --bind 127.0.0.1 \
  --directory /home/milab/cvpr2026cctv/team01/minseok/qwen_tv/dataset \
  > video_http_8002.log 2>&1 &
```

* 추론 돌리기

모델 자체 모듈에서의 기본값 fps: 2.0
```bash
CUDA_VISIBLE_DEVICES=0,1 nohup python3 inference_qwen35_openai_vllm_classification.py \
  --model_id /home/milab/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B-FP8/snapshots/2e1b21350ce589fcaafbb3c7d7eac526a7aed582 \
  --openai_base_url http://127.0.0.1:8000/v1 \
  --openai_api_key EMPTY \
  --base_video_url http://127.0.0.1:8002 \
  --base_submission_csv submission_rian.csv \
  --max_tokens 256 \
  --experiment_name qwen35_27b_vllm_replcae_only_classification_from_rian \
  > new_prompt_27b.log 2>&1 &
disown
```