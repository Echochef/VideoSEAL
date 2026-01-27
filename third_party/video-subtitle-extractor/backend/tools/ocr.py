import os
import importlib
import paddleocr as paddleocr_pkg
from paddleocr import PaddleOCR
import config

# 加载文本检测+识别模型
class OcrRecogniser:
    def __init__(self):
        # 获取参数对象
        importlib.reload(config)
        self._use_paddlex_api = self._should_use_paddlex_api()
        self.recogniser = self.init_model()

    @staticmethod
    def _should_use_paddlex_api():
        """Return True when PaddleOCR>=3.0 (new PaddleX pipeline)."""
        version_str = getattr(paddleocr_pkg, "__version__", "0")
        try:
            major = int(version_str.split(".")[0])
            return major >= 3
        except (ValueError, AttributeError):
            return False

    @staticmethod
    def _parse_rec_image_shape(raw_shape):
        """Convert comma separated string like '3,48,320' to tuple expected by new API."""
        if isinstance(raw_shape, (list, tuple)) and len(raw_shape) == 3:
            try:
                return tuple(int(v) for v in raw_shape)
            except (TypeError, ValueError):
                return None
        if isinstance(raw_shape, str):
            parts = [p.strip() for p in raw_shape.split(',') if p.strip()]
            if len(parts) == 3:
                try:
                    return tuple(int(p) for p in parts)
                except ValueError:
                    return None
        return None

    @staticmethod
    def _has_paddlex_config(model_dir):
        if not model_dir:
            return False
        config_path = os.path.join(model_dir, "inference.yml")
        return os.path.isfile(config_path)

    def _predict_with_paddlex(self, image):
        """Run inference via PaddleOCR>=3.0 pipeline and mimic legacy return format."""
        results = self.recogniser.predict(image)
        detection_box = []
        recognise_result = []
        if not results:
            return detection_box, recognise_result

        for item in results:
            polys = item.get('rec_polys') or item.get('dt_polys') or []
            boxes = item.get('rec_boxes')
            texts = item.get('rec_texts') or []
            scores = item.get('rec_scores') or []
            for idx, text in enumerate(texts):
                score = float(scores[idx]) if idx < len(scores) else 0.0
                if idx < len(polys) and polys[idx] is not None:
                    poly = polys[idx]
                    try:
                        pts = poly.tolist()
                    except AttributeError:
                        pts = poly
                    points = [(float(pt[0]), float(pt[1])) for pt in pts]
                elif boxes is not None and idx < len(boxes):
                    box = boxes[idx]
                    points = [
                        (float(box[0]), float(box[1])),
                        (float(box[2]), float(box[1])),
                        (float(box[2]), float(box[3])),
                        (float(box[0]), float(box[3]))
                    ]
                else:
                    points = [(0.0, 0.0)] * 4
                detection_box.append(points)
                recognise_result.append((text, score))

        return detection_box, recognise_result

    @staticmethod
    def y_round(y):
        y_min = y + 10 - y % 10
        y_max = y - y % 10
        if abs(y - y_min) < abs(y - y_max):
            return y_min
        else:
            return y_max

    def predict(self, image):
        if self._use_paddlex_api:
            detection_box, recognise_result = self._predict_with_paddlex(image)
        else:
            detection_box, recognise_result, _ = self.recogniser(image, cls=False)
        if len(detection_box) > 0:
            coordinate_list = list()
            if isinstance(detection_box, list):
                for i in detection_box:
                    i = list(i)
                    (x1, y1) = int(i[0][0]), int(i[0][1])
                    (x2, y2) = int(i[1][0]), int(i[1][1])
                    (x3, y3) = int(i[2][0]), int(i[2][1])
                    (x4, y4) = int(i[3][0]), int(i[3][1])
                    xmin = max(x1, x4)
                    xmax = min(x2, x3)
                    ymin = max(y1, y2)
                    ymax = min(y3, y4)
                    coordinate_list.append([xmin, xmax, ymin, ymax])

            # 计算有多少行字幕，将每行字幕最小的ymin值放入lines
            lines = []
            for i in coordinate_list:
                if len(lines) < 1:
                    lines.append(self.y_round(i[2]))
                else:
                    if self.y_round(i[2]) not in lines \
                            and self.y_round(i[2]) + 10 not in lines \
                            and self.y_round(i[2]) - 10 not in lines:
                        lines.append(self.y_round(i[2]))
            lines = sorted(lines)

            for i in coordinate_list:
                for j in lines:
                    if abs(j - self.y_round(i[2])) <= 10:
                        i[2] = j

            to_rank_res = list(zip(coordinate_list, recognise_result))
            ranked_res = []
            for line in lines:
                tmp_list = []
                for i in to_rank_res:
                    if i[0][2] == line:
                        tmp_list.append(i)
                # 先根据纵坐标排序
                for k in range(1, len(tmp_list)):
                    for j in range(0, len(tmp_list) - k):
                        if tmp_list[j][0][2] > tmp_list[j + 1][0][2]:
                            print(tmp_list[j][0][2])
                            tmp_list[j], tmp_list[j + 1] = tmp_list[j + 1], tmp_list[j]
                # 再根据横坐标排列
                for l in range(1, len(tmp_list)):
                    for j in range(0, len(tmp_list) - l):
                        if tmp_list[j][0][0] > tmp_list[j + 1][0][0]:
                            tmp_list[j], tmp_list[j + 1] = tmp_list[j + 1], tmp_list[j]
                for m in tmp_list:
                    ranked_res.append(m)
            dt_box = []
            for i in [j[0] for j in ranked_res]:
                dt_box.append([(i[0], i[2]), (i[1], i[2]), (i[1], i[3]), (i[0], i[3])])
            res = [i[1] for i in ranked_res]
            return dt_box, res
        else:
            return detection_box, recognise_result

    def init_model(self):
        if self._use_paddlex_api:
            if config.ONNX_PROVIDERS:
                # PaddleOCR>=3.0 switches to PaddleX pipeline which no longer supports ONNX
                print("ONNX execution providers are configured but PaddleOCR>=3.0 only supports Paddle models; falling back to Paddle inference.")
            rec_shape = self._parse_rec_image_shape(config.REC_IMAGE_SHAPE)
            det_model_dir = config.DET_MODEL_PATH if self._has_paddlex_config(config.DET_MODEL_PATH) else None
            rec_model_dir = config.REC_MODEL_PATH if self._has_paddlex_config(config.REC_MODEL_PATH) else None
            if det_model_dir is None or rec_model_dir is None:
                print("PaddleOCR>=3.0 detected but local OCR weights are not PaddleX exports (missing inference.yml); falling back to built-in pretrained models.")
            params = dict(
                device="gpu" if config.USE_GPU else "cpu",
                text_recognition_batch_size=config.REC_BATCH_NUM,
                lang=config.REC_CHAR_TYPE,
                ocr_version=f'PP-OCR{config.MODEL_VERSION.lower()}',
                text_rec_score_thresh=0,
            )
            if det_model_dir:
                params["text_detection_model_dir"] = det_model_dir
            if rec_model_dir:
                params["text_recognition_model_dir"] = rec_model_dir
            if rec_shape:
                params["text_rec_input_shape"] = rec_shape
            return PaddleOCR(**params)

        det_model_dir = self.convertToOnnxModelIfNeeded(config.DET_MODEL_PATH)
        rec_model_dir = self.convertToOnnxModelIfNeeded(config.REC_MODEL_PATH)
        return PaddleOCR(use_gpu=config.USE_GPU,
                         gpu_mem=500,
                         det_algorithm='DB',
                         det_model_dir=det_model_dir,
                         rec_algorithm='CRNN',
                         rec_batch_num=config.REC_BATCH_NUM,
                         rec_model_dir=rec_model_dir,
                         max_batch_size=config.MAX_BATCH_SIZE,
                         det=True,
                         use_angle_cls=False,
                         drop_score=0,
                         lang=config.REC_CHAR_TYPE,
                         ocr_version=f'PP-OCR{config.MODEL_VERSION.lower()}',
                         rec_image_shape=config.REC_IMAGE_SHAPE,
                         use_onnx=len(config.ONNX_PROVIDERS) > 0,
                         onnx_providers=config.ONNX_PROVIDERS,
                         debug=False)
    

    def convertToOnnxModelIfNeeded(self, model_dir, model_filename="inference.pdmodel", params_filename="inference.pdiparams", opset_version=14):
        """Converts a Paddle model to ONNX if ONNX providers are available and the model does not already exist."""
        
        if not config.ONNX_PROVIDERS:
            return model_dir
        
        onnx_model_path = os.path.join(model_dir, "model.onnx")

        if os.path.exists(onnx_model_path):
            print(f"ONNX model already exists: {onnx_model_path}. Skipping conversion.")
            return onnx_model_path
        
        print(f"Converting Paddle model {model_dir} to ONNX...")
        model_file = os.path.join(model_dir, model_filename)
        params_file = os.path.join(model_dir, params_filename) if params_filename else ""

        try:
            import paddle2onnx
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)

            # Convert and save the model
            onnx_model = paddle2onnx.export(
                model_filename=model_file,
                params_filename=params_file,
                save_file=onnx_model_path,
                opset_version=opset_version,
                auto_upgrade_opset=True,
                verbose=True,
                enable_onnx_checker=True,
                enable_experimental_op=True,
                enable_optimize=True,
                custom_op_info={},
                deploy_backend="onnxruntime",
                calibration_file="calibration.cache",
                external_file=os.path.join(model_dir, "external_data"),
                export_fp16_model=False,
            )

            print(f"Conversion successful. ONNX model saved to: {onnx_model_path}")
            return onnx_model_path
        except Exception as e:
            print(f"Error during conversion: {e}")
            return model_dir


def get_coordinates(dt_box):
    """
    从返回的检测框中获取坐标
    :param dt_box 检测框返回结果
    :return list 坐标点列表
    """
    coordinate_list = list()
    if isinstance(dt_box, list):
        for i in dt_box:
            try:
                i = list(i)
                # 容错：确保检测框包含4个点、每点含2个坐标
                if len(i) < 4:
                    continue
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = max(x1, x4)
                xmax = min(x2, x3)
                ymin = max(y1, y2)
                ymax = min(y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
            except Exception:
                # 非标准返回（例如缺点/空框）直接跳过
                continue
    return coordinate_list
