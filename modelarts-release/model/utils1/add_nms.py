import numpy as np
import onnx
from onnx import shape_inference
try:
    import onnx_graphsurgeon as gs
except Exception as e:
    print('Import onnx_graphsurgeon failure: %s' % e)

import logging

LOGGER = logging.getLogger(__name__)


class RegisterNMS(object):
    def __init__(self, onnx_model_path: str, precision: str = "fp32"):

        self.graph = gs.import_onnx(onnx.load(onnx_model_path))
        assert self.graph
        LOGGER.info("ONNX graph created successfully")
        self.graph.fold_constants()
        self.precision = precision
        self.batch_size = 1

    def infer(self):
        for _ in range(3):
            count_before = len(self.graph.nodes)
            self.graph.cleanup().toposort()
            try:
                for node in self.graph.nodes:
                    for o in node.outputs:
                        o.shape = None
                model = gs.export_onnx(self.graph)
                model = shape_inference.infer_shapes(model)
                self.graph = gs.import_onnx(model)
            except Exception as e:
                LOGGER.info(f"Shape inference could not be performed at this time:\n{e}")
            try:
                self.graph.fold_constants(fold_shapes=True)
            except TypeError as e:
                LOGGER.error(
                    "This version of ONNX GraphSurgeon does not support folding shapes, "
                    f"please upgrade your onnx_graphsurgeon module. Error:\n{e}"
                )
                raise

            count_after = len(self.graph.nodes)
            if count_before == count_after:
                break

    def save(self, output_path):
        self.graph.cleanup().toposort()
        model = gs.export_onnx(self.graph)
        onnx.save(model, output_path)
        LOGGER.info(f"Saved ONNX model to {output_path}")

    def register_nms(self, *, score_thresh: float = 0.25, nms_thresh: float = 0.45, detections_per_img: int = 100):
        self.infer()
        op_inputs = self.graph.outputs
        op = "EfficientNMS_TRT"
        attrs = {
            "plugin_version": "1",
            "background_class": -1,
            "max_output_boxes": detections_per_img,
            "score_threshold": score_thresh,
            "iou_threshold": nms_thresh,
            "score_activation": False,
            "box_coding": 0,
        }

        if self.precision == "fp32":
            dtype_output = np.float32
        elif self.precision == "fp16":
            dtype_output = np.float16
        else:
            raise NotImplementedError(f"Currently not supports precision: {self.precision}")

        output_num_detections = gs.Variable(
            name="num_dets",
            dtype=np.int32,
            shape=[self.batch_size, 1],
        )
        output_boxes = gs.Variable(
            name="det_boxes",
            dtype=dtype_output,
            shape=[self.batch_size, detections_per_img, 4],
        )
        output_scores = gs.Variable(
            name="det_scores",
            dtype=dtype_output,
            shape=[self.batch_size, detections_per_img],
        )
        output_labels = gs.Variable(
            name="det_classes",
            dtype=np.int32,
            shape=[self.batch_size, detections_per_img],
        )

        op_outputs = [output_num_detections, output_boxes, output_scores, output_labels]

        self.graph.layer(op=op, name="batched_nms", inputs=op_inputs, outputs=op_outputs, attrs=attrs)
        LOGGER.info(f"Created NMS plugin '{op}' with attributes: {attrs}")

        self.graph.outputs = op_outputs

        self.infer()

    def save(self, output_path):
        self.graph.cleanup().toposort()
        model = gs.export_onnx(self.graph)
        onnx.save(model, output_path)
        LOGGER.info(f"Saved ONNX model to {output_path}")
