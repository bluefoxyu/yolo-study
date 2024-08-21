package com.bluefoxyu.config;

import ai.onnxruntime.OrtException;
import com.bluefoxyu.model.YoloV7;
import com.bluefoxyu.model.YoloV8;
import com.bluefoxyu.model.domain.Onnx;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class OnnxConfig {

    static String yolov8_model_path = "./dp/src/main/resources/model/yolov8s.onnx";
    static String yolov7_model_path = "./dp/src/main/resources/model/yolov7-tiny.onnx";

    static String[] names = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
            "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
            "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
            "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"};

    @Bean(name = "YoloV8Onnx")
    public Onnx YoloV8Onnx() throws OrtException {
        // 加载模型(按需求修改需要加载的模型)
        String modelPath = yolov8_model_path;
        return new YoloV8(names, modelPath, false);
    }

    @Bean(name = "YoloV7Onnx")
    public Onnx YoloV7Onnx() throws OrtException {
        // 加载模型(按需求修改需要加载的模型)
        String modelPath = yolov7_model_path;
        return new YoloV7(names, modelPath, false);
    }

}
