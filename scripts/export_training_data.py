"""검토 큐 교정 데이터 → Fine-tuning 학습 데이터 변환

검토 큐에서 승인(approved) 완료된 교정 데이터를 JSONL 형식으로 export합니다.

출력:
  data/training/vlm_sft/corrected_{timestamp}.jsonl   — SFT용
  data/training/vlm_dpo/corrected_{timestamp}.jsonl   — DPO용 (chosen/rejected)

실행:
  python scripts/export_training_data.py [--db-url sqlite:///data/review_queue.db]
  python scripts/export_training_data.py --format dpo
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.postprocess.review_queue import ReviewQueue, ReviewQueueConfig


def main():
    parser = argparse.ArgumentParser(description="검토 큐 → 학습 데이터 JSONL 변환")
    parser.add_argument("--db-url", default="sqlite:///data/review_queue.db")
    parser.add_argument("--format", choices=["sft", "dpo"], default="sft")
    parser.add_argument("--status", default="approved")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    rq = ReviewQueue(ReviewQueueConfig(db_url=args.db_url))
    pairs = rq.export_training_pairs(status=args.status, output_format=args.format)

    if not pairs:
        print(f"export 대상 없음 (status={args.status})")
        return

    # 출력 경로
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.format == "dpo":
        out_dir = _ROOT / "data" / "training" / "vlm_dpo"
    else:
        out_dir = _ROOT / "data" / "training" / "vlm_sft"

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"corrected_{timestamp}.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"export 완료: {len(pairs)}건 → {out_path}")


if __name__ == "__main__":
    main()
