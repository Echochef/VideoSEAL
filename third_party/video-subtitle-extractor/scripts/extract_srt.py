#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin runner: parse CLI args and call backend.main.extract_srt.
"""
import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path so that `import backend` works
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import extract_srt


def main():
    parser = argparse.ArgumentParser(description='视频字幕提取（生成 SRT）')
    parser.add_argument('--video', required=True, help='视频绝对或相对路径')
    parser.add_argument('--language', default='en', help='OCR 语言代码（如 ch/en/japan/korean 等）')
    parser.add_argument('--mode', default='auto', choices=['fast', 'auto', 'accurate'], help='模式（默认 auto）')
    parser.add_argument('--area', nargs='+', help='字幕区域 (ymin ymax xmin xmax) 或逗号分隔字符串')
    parser.add_argument('--drop-score', type=float, default=None, help='覆盖识别置信度阈值（默认不覆盖）')
    parser.add_argument('--sub-area-deviation-rate', type=float, default=None, help='区域越界容忍比例，建议0.05~0.2')
    parser.add_argument('--no-word-seg', action='store_true', help='关闭英文分词/清洗（默认开启）')
    parser.add_argument('--gen-txt', action='store_true', help='额外生成 .txt（默认否）')
    parser.add_argument('--keep-empty-ts', action='store_true', help='当使用 VSF 合并时保留空时间轴（默认不保留）')
    parser.add_argument('--interface', default='English', help='仅影响日志语言，默认 English')
    parser.add_argument('--vsf-use-bottom-default', default='true', help='未传 area 时是否使用底部区域（true/false），默认 true')
    parser.add_argument('--vsf-bottom-ratio', type=float, default=0.25, help='底部区域占比，默认 0.25')
    parser.add_argument('--output-dir', default=None, help='自定义输出目录（SRT/TXT）')
    parser.add_argument('--filter-by-area', default='true', help='是否按区域过滤文本（true/false），默认 true')
    parser.add_argument('--split-regions', default='true', help='是否按区域拆分（字幕/辅助）写入（true/false），默认 true；false=整图统一写入')

    args = parser.parse_args()

    def parse_area(arglist):
        if not arglist:
            return None
        if len(arglist) == 1:
            token = arglist[0].replace(',', ' ').strip()
            parts = [p for p in token.split() if p]
        else:
            parts = arglist
        if len(parts) != 4:
            parser.error('--area 需要 4 个整数：ymin ymax xmin xmax')
        try:
            ymin, ymax, xmin, xmax = map(int, parts)
            return (ymin, ymax, xmin, xmax)
        except Exception:
            parser.error('--area 需要 4 个整数：ymin ymax xmin xmax')

    def to_bool(s: str) -> bool:
        return str(s).strip().lower() in ('1', 'true', 'yes', 'y')

    area = parse_area(args.area)
    def to_bool(s: str) -> bool:
        return str(s).strip().lower() in ('1', 'true', 'yes', 'y')

    srt_path = extract_srt(
        args.video,
        language=args.language,
        mode=args.mode,
        area=area,
        drop_score=args.drop_score,
        sub_area_deviation_rate=args.sub_area_deviation_rate,
        word_segmentation=(not args.no_word_seg),
        generate_txt=args.gen_txt,
        keep_empty_ts=args.keep_empty_ts,
        interface=args.interface,
        vsf_use_bottom_default=to_bool(args.vsf_use_bottom_default),
        vsf_bottom_ratio=args.vsf_bottom_ratio,
        output_dir=args.output_dir,
        filter_by_area=to_bool(args.filter_by_area),
        split_regions=to_bool(args.split_regions),
    )
    print(srt_path)


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'[Error] {type(e).__name__}: {e}', file=sys.stderr)
        sys.exit(2)
