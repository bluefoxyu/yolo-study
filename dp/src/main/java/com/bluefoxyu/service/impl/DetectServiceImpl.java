package com.bluefoxyu.service.impl;

import ai.onnxruntime.OrtException;
import com.bluefoxyu.model.domain.Onnx;
import com.bluefoxyu.output.Output;
import com.bluefoxyu.service.DetectService;
import jakarta.annotation.Resource;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class DetectServiceImpl implements DetectService {

    @Resource
    @Qualifier("YoloV8Onnx") //指定注入的 Bean 是 OnnxConfig 类中由 @Bean 注解生成的、名称为 "YoloV8Onnx" 的 Bean。
    private Onnx yoloV8Onnx;

    @Resource
    @Qualifier("YoloV7Onnx") //指定注入的 Bean 是 OnnxConfig 类中由 @Bean 注解生成的、名称为 "YoloV8Onnx" 的 Bean。
    private Onnx yoloV7Onnx;

    @Override
    public List<Output> yoloV8Detection(String test_img) throws OrtException {

        // 1. 初始化模型
        // 全局new一次即可，千万不要每次使用都new。可以使用@Bean，或者在spring项目启动时初始化一次即可
        /*Onnx onnx = new YoloV8(names,model_path,false);*/

        // 2. 读取图像
        // 也可以使用接口收到的base64图像Imgcodecs.imdecode()
        Mat img = Imgcodecs.imread(test_img);

        // 3. 执行模型推理
        // 这一步已经结束，可以通过接口返回给前端结果，或者自己循环打印看结果输出
        List<Output> outputs = yoloV8Onnx.run(img.clone());

        // 4. 处理并保存图像
        // 可以调用此方法本地查看图片效果，也可以不调用
        yoloV8Onnx.drawprocess(outputs,img);

        return outputs;

    }

    @Override
    public List<Output> yoloV7Detection(String test_img) throws OrtException {

        // 1. 初始化模型
        // 全局new一次即可，千万不要每次使用都new。可以使用@Bean，或者在spring项目启动时初始化一次即可
        /*Onnx onnx = new YoloV8(names,model_path,false);*/

        // 2. 读取图像
        // 也可以使用接口收到的base64图像Imgcodecs.imdecode()
        Mat img = Imgcodecs.imread(test_img);

        // 3. 执行模型推理
        // 这一步已经结束，可以通过接口返回给前端结果，或者自己循环打印看结果输出
        List<Output> outputs = yoloV7Onnx.run(img.clone());

        // 4. 处理并保存图像
        // 可以调用此方法本地查看图片效果，也可以不调用
        yoloV7Onnx.drawprocess(outputs,img);

        return outputs;

    }
}
