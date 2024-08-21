package com.bluefoxyu.demo;

import ai.onnxruntime.OrtException;
import com.bluefoxyu.model.YoloV8;
import com.bluefoxyu.model.domain.Onnx;
import com.bluefoxyu.output.Output;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.List;

public class Main {

   static String model_path = "./dp/src/main/resources/model/yolov8s.onnx";

   static String test_img = "./dp/images/some_people.png";

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

    public static void main(String[] args) throws OrtException {

        // 1. 初始化模型
        // 全局new一次即可，千万不要每次使用都new。可以使用@Bean，或者在spring项目启动时初始化一次即可
        Onnx onnx = new YoloV8(names,model_path,false);
        //Onnx onnx = new YoloV7(names,model_path,false);
        //Onnx onnx = new YoloV5(labels,model_path,false);

        // 2. 读取图像
        // 也可以使用接口收到的base64图像Imgcodecs.imdecode()
        Mat img = Imgcodecs.imread(test_img);

        // 3. 执行模型推理
        // 这一步已经结束，可以通过接口返回给前端结果，或者自己循环打印看结果输出
        List<Output> outputs = onnx.run(img.clone());

        // 4. 处理并保存图像
        // 可以调用此方法本地查看图片效果，也可以不调用
        onnx.drawprocess(outputs,img);

    }
}
