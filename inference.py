import os
import ast
import json
import time
import multiprocessing as mp
from datetime import datetime

import torch
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForImageTextToText


# .env에 HF_TOKEN이 있으면 Hugging Face 로그인까지 자동으로 수행한다.
# 공개 모델만 쓸 때는 토큰이 없어도 import 단계에서 죽지 않도록 처리한다.
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("HF_TOKEN is not set. Continuing without Hugging Face login.", flush=True)


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8b-Instruct"
DEFAULT_TEST_PATH_FILE = "dataset/test_video_path.txt"
DEFAULT_EXPERIMENT_NAME = "qwen3_vl_inference"


def parse_in_json(llm_response, video_path):
    # 모델 응답을 Python literal/JSON 형태로 파싱하고,
    # 실패 시에도 submission 생성을 이어갈 수 있도록 기본값을 넣는다.
    try:
        temp = ast.literal_eval(llm_response)
    except Exception:
        temp = {
            "time": 0.0,
            "coordinate": [[0, 0], [0, 0]],
            "type": "head-on",
            "why": "invalid response format"
        }

    if not isinstance(temp, dict):
        temp = {
            "time": 0.0,
            "coordinate": [[0, 0], [0, 0]],
            "type": "head-on",
            "why": "invalid response format"
        }

    save_dir = os.path.join("result", "parsed_json")
    os.makedirs(save_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(save_dir, f"{video_name}.json")
    # 원문 응답을 바로 쓰지 않고, 후처리된 JSON을 파일로 저장해
    # 나중에 샘플별 디버깅과 오류 확인에 재사용한다.
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(temp, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"failed to save parsed json for {video_path}: {e}", flush=True)

    return temp


def save_submission(submission, experiment_name, description_text=""):
    # 실행 시각 기준으로 run 디렉터리를 만들고,
    # submission.csv와 실행 메모(description.txt)를 함께 저장한다.
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = os.path.join("result", f"{experiment_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    submission_path = os.path.join(save_dir, "submission.csv")
    description_path = os.path.join(save_dir, "description.txt")

    submission.to_csv(submission_path, index=False, lineterminator="\n")

    with open(description_path, "w", encoding="utf-8") as f:
        f.write(description_text)

    print(f"saved to {save_dir}")


def make_submission(results):
    # 파싱된 추론 결과를 대회 제출 포맷(path, time, center_x/y, type)으로 변환한다.
    # 좌표는 bbox 중심점을 0~1 스케일로 맞추기 위해 1000으로 나눠 저장한다.
    rows = []

    for result in results:
        try:
            video_path = result.get("video_path", "")
            path = "videos/" + video_path.split("/videos/")[-1] if "/videos/" in video_path else video_path

            accident_time = result.get("time", 0.0)
            coordinate = result.get("coordinate", [[0, 0], [0, 0]])
            (x1, y1), (x2, y2) = coordinate

            center_x = round(((x1 + x2) / 2) / 1000, 3)
            center_y = round(((y1 + y2) / 2) / 1000, 3)
            accident_type = result.get("type", "unknown")

            rows.append({
                "path": path,
                "accident_time": round(accident_time, 2),
                "center_x": center_x,
                "center_y": center_y,
                "type": accident_type
            })
        except Exception as e:
            print(f"failed to make submission for result: {result}, error: {e}")

    submission = pd.DataFrame(
        rows,
        columns=["path", "accident_time", "center_x", "center_y", "type"]
    )
    return submission


class VideoInferenceVLM:
    # 다른 비디오-언어 모델 구현체로 교체할 수 있게 둔 최소 인터페이스다.
    def video_inference(self, video_path, prompt, max_new_tokens=128):
        raise NotImplementedError("Subclasses should implement this method.")


class Qwen3VLInference(VideoInferenceVLM):
    def __init__(self, model_id=DEFAULT_MODEL_ID, device="cuda:0"):
        # 모델과 processor를 프로세스별로 한 번만 올리고,
        # 이후에는 같은 GPU에서 여러 비디오를 순차 처리한다.
        self.device = device
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype="auto",
            device_map={"": device}
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def video_inference(self, video_path, prompt, max_new_tokens=128):
        # Qwen3-VL chat template 형식에 맞춰
        # 비디오 1개 + 텍스트 프롬프트를 하나의 대화 입력으로 구성한다.
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                    },
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }
        ]

        # processor가 멀티모달 입력을 토큰화/텐서화하고,
        # 최종 텐서는 현재 워커의 GPU로 이동시킨다.
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_text


def worker_process(gpu_id, start_idx, end_idx, video_paths, prompt, output_queue):
    # 워커 1개가 GPU 1개를 전담한다.
    # 자신에게 할당된 video index 구간만 처리한 뒤, 결과를 queue로 메인 프로세스에 돌려준다.
    try:
        torch.cuda.set_device(gpu_id)
        inference = Qwen3VLInference(device=f"cuda:{gpu_id}")
        results = []

        loop_start_time = time.time()

        for i in range(start_idx, end_idx):
            video_path = video_paths[i]
            output = inference.video_inference(
                video_path,
                prompt,
                max_new_tokens=128
            )
            output_json = parse_in_json(output[0], video_path)
            output_json["video_path"] = video_path
            results.append((i, output_json))
            print(f"gpu {gpu_id} - {i + 1}/{len(video_paths)} is done.", flush=True)

        loop_end_time = time.time()
        elapsed_seconds = loop_end_time - loop_start_time

        output_queue.put({
            "gpu_id": gpu_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "elapsed_seconds": elapsed_seconds,
            "results": results,
            "error": None
        })
    except Exception as e:
        output_queue.put({
            "gpu_id": gpu_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "elapsed_seconds": 0.0,
            "results": [],
            "error": repr(e)
        })


def main():
    # CUDA + transformers 멀티프로세싱 안정성을 위해 spawn 방식을 강제한다.
    mp.set_start_method("spawn", force=True)

    # 프롬프트는 충돌 시각, bbox, 사고 유형을 JSON 하나로 반환하도록 고정한다.
    # 이 파일은 CoT를 길게 쓰기보다 제출 포맷에 맞는 구조화 출력에 집중한다.
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

    test_path_file = DEFAULT_TEST_PATH_FILE
    prompt = instruction + return_format

    # test_video_path.txt에 적힌 비디오 경로를 모두 읽는다.
    with open(test_path_file, "r", encoding="utf-8") as f:
        video_paths = [line.strip() for line in f if line.strip()]

    total_videos = len(video_paths)
    if total_videos == 0:
        raise ValueError(f"No videos found in {test_path_file}")

    # 보이는 GPU 수만큼 worker를 만들되,
    # 비디오 개수보다 worker가 많아지는 경우는 피한다.
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count <= 0:
        raise RuntimeError("No CUDA device is visible. inference.py requires at least one GPU.")

    worker_count = min(visible_gpu_count, total_videos)
    chunk_size = (total_videos + worker_count - 1) // worker_count

    output_queue = mp.Queue()
    processes = []

    # 전체 비디오 목록을 contiguous chunk로 나눠 worker에 배정한다.
    for worker_index in range(worker_count):
        start_idx = worker_index * chunk_size
        end_idx = min(total_videos, start_idx + chunk_size)
        if start_idx >= end_idx:
            continue

        process = mp.Process(
            target=worker_process,
            args=(worker_index, start_idx, end_idx, video_paths, prompt, output_queue)
        )
        processes.append(process)

    total_start_time = time.time()

    for process in processes:
        process.start()

    # 각 워커가 queue에 넣은 요약 결과를 수집한다.
    worker_outputs = [output_queue.get() for _ in processes]

    for process in processes:
        process.join()

    total_end_time = time.time()
    total_elapsed_seconds = total_end_time - total_start_time

    ordered_results = []
    worker_errors = []
    worker_summaries = []
    for worker_output in worker_outputs:
        worker_summaries.append(
            f"gpu={worker_output['gpu_id']} range=[{worker_output['start_idx']}, {worker_output['end_idx']}) "
            f"elapsed={worker_output['elapsed_seconds']:.2f}s error={worker_output['error']}"
        )
        if worker_output["error"] is not None:
            worker_errors.append(
                f"gpu {worker_output['gpu_id']} failed for range "
                f"[{worker_output['start_idx']}, {worker_output['end_idx']}): {worker_output['error']}"
            )
        ordered_results.extend(worker_output["results"])

    # 멀티프로세싱 결과는 순서가 섞일 수 있으므로 원래 video index 기준으로 다시 정렬한다.
    ordered_results.sort(key=lambda item: item[0])
    parsed_results = [payload for _, payload in ordered_results]
    submission = make_submission(parsed_results)

    # 실행 메타데이터를 description.txt에 남겨
    # 어떤 환경과 분배 전략으로 돌렸는지 추적할 수 있게 한다.
    description_lines = [
        f"model_id: {DEFAULT_MODEL_ID}",
        f"test_path_file: {test_path_file}",
        f"visible_gpu_count: {visible_gpu_count}",
        f"worker_count: {worker_count}",
        f"total_videos: {total_videos}",
        f"total_elapsed_seconds: {total_elapsed_seconds:.2f}",
        "worker_outputs:",
    ]
    description_lines.extend(worker_summaries)
    if worker_errors:
        description_lines.append("errors:")
        description_lines.extend(worker_errors)

    save_submission(
        submission,
        experiment_name=DEFAULT_EXPERIMENT_NAME,
        description_text="\n".join(description_lines) + "\n",
    )

    print(f"Total elapsed time: {total_elapsed_seconds:.2f} seconds")
    if worker_errors:
        print("Some worker processes failed:", flush=True)
        for error_text in worker_errors:
            print(error_text, flush=True)


if __name__ == "__main__":
    main()
