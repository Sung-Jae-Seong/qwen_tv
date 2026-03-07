import os
import ast
import time
import multiprocessing as mp
from datetime import datetime

import torch
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForImageTextToText


load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)


def parse_in_json(llm_response):
    try:
        temp = ast.literal_eval(llm_response)
    except Exception:
        return {
            "time": 0.0,
            "coordinate": [[0, 0], [0, 0]],
            "type": "head-on",
            "why": "invalid response format"
        }

    if not isinstance(temp, dict):
        return {
            "time": 0.0,
            "coordinate": [[0, 0], [0, 0]],
            "type": "head-on",
            "why": "invalid response format"
        }

    return temp


def save_submission(submission, experiment_name, description_text=""):
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
    rows = []

    for result in results:
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

    submission = pd.DataFrame(
        rows,
        columns=["path", "accident_time", "center_x", "center_y", "type"]
    )
    return submission


class VideoInferenceVLM:
    def video_inference(self, video_path, prompt, max_new_tokens=128):
        raise NotImplementedError("Subclasses should implement this method.")


class Qwen3VLInference(VideoInferenceVLM):
    def __init__(self, model_id="Qwen/Qwen3-VL-8b-Instruct", device="cuda:0"):
        self.device = device
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype="auto",
            device_map={"": device}
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def video_inference(self, video_path, prompt, max_new_tokens=128):
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
            output_json = parse_in_json(output[0])
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
    mp.set_start_method("spawn", force=True)

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

    test_path_file = "dataset/test_video_path.txt"
    prompt = instruction + return_format

    with open(test_path_file, "r", encoding="utf-8") as f:
        video_paths = [line.strip() for line in f if line.strip()]

    total_videos = len(video_paths)
    mid = total_videos // 2

    output_queue = mp.Queue()

    process_0 = mp.Process(
        target=worker_process,
        args=(0, 0, mid, video_paths, prompt, output_queue)
    )
    process_1 = mp.Process(
        target=worker_process,
        args=(1, mid, total_videos, video_paths, prompt, output_queue)
    )

    total_start_time = time.time()

    process_0.start()
    process_1.start()

    worker_outputs = [output_queue.get(), output_queue.get()]

    process_0.join()
    process_1.join()

    total_end_time = time.time()
    total_elapsed_seconds = total_end_time - total_start_time

    errors = [x for x in worker_outputs if x["error"] is not None]
    if len(errors) > 0:
        raise RuntimeError(str(errors))

    merged_results = []
    for worker_output in worker_outputs:
        merged_results.extend(worker_output["results"])

    merged_results.sort(key=lambda x: x[0])
    final_results = [result for _, result in merged_results]

    submission = make_submission(final_results)

    description_lines = [
        f"total_videos: {total_videos}",
        f"total_elapsed_seconds: {total_elapsed_seconds:.4f}"
    ]

    for worker_output in sorted(worker_outputs, key=lambda x: x["gpu_id"]):
        description_lines.append(
            f"gpu{worker_output['gpu_id']}_elapsed_seconds: {worker_output['elapsed_seconds']:.4f}"
        )

    description_text = "\n".join(description_lines) + "\n"

    save_submission(submission, "qwen3_vl_experiment_2gpu", description_text)


if __name__ == "__main__":
    main()