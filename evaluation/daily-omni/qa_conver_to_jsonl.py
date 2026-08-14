import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="原始qa.json完整路径")
    args = parser.parse_args()

    src = args.src
    # 输出和qa.json同目录
    out_dir = os.path.dirname(src)
    output_jsonl = os.path.join(out_dir, "daily_omni.jsonl")

    with open(src,"r",encoding="utf-8") as f:
        raw = json.load(f)

    out_lines = []
    for item in raw:
        vid = item["video_id"]
        converted = {
            "video_id": vid,
            "question": item["Question"],
            "choices": item["Choice"],
            "gt_answer": item["Answer"],
            "VideoPath": f"Videos/{vid}/{vid}_video.mp4",
            "WavPath": f"Videos/{vid}/{vid}_audio.wav",
            "qa_type": item.get("Type",""),
            "content_parent_category": item.get("content_parent_category",""),
            "content_fine_category": item.get("content_fine_category",""),
            "video_category": item.get("video_category",""),
            "video_duration": item.get("video_duration",""),
        }
        out_lines.append(json.dumps(converted, ensure_ascii=False))

    with open(output_jsonl,"w",encoding="utf-8") as f:
        f.write("\n".join(out_lines))

    print(f"输入：{src}")
    print(f"输出：{output_jsonl}")
    print(f"共生成 {len(out_lines)} 条样本")

if __name__ == "__main__":
    main()
