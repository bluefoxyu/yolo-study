package com.bluefoxyu.service.impl;

import com.bluefoxyu.model.domain.Onnx;
import com.bluefoxyu.output.Output;
import com.bluefoxyu.service.DetectService;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.List;

@Slf4j
@Service
public class DetectServiceImpl implements DetectService {

    @Resource
    @Qualifier("YoloV8Onnx") //指定注入的 Bean 是 OnnxConfig 类中由 @Bean 注解生成的、名称为 "YoloV8Onnx" 的 Bean。
    private Onnx yoloV8Onnx;

    @Resource
    @Qualifier("YoloV7Onnx") //指定注入的 Bean 是 OnnxConfig 类中由 @Bean 注解生成的、名称为 "YoloV8Onnx" 的 Bean。
    private Onnx yoloV7Onnx;

    @Override
    public List<Output> yoloV8Detection(String test_img) throws Exception {

        // 1. 初始化模型
        // 全局new一次即可，千万不要每次使用都new。可以使用@Bean，或者在spring项目启动时初始化一次即可
        /*Onnx onnx = new YoloV8(names,model_path,false);*/

        // 2. 读取图像
        // 也可以使用接口收到的base64图像Imgcodecs.imdecode()
        Mat img = readOrDownloadImage(test_img);

        // 3. 执行模型推理
        // 这一步已经结束，可以通过接口返回给前端结果，或者自己循环打印看结果输出
        List<Output> outputs = yoloV8Onnx.run(img.clone());

        // 4. 处理并保存图像
        // 可以调用此方法本地查看图片效果，也可以不调用
        yoloV8Onnx.drawprocess(outputs,img);

        return outputs;

    }

    @Override
    public List<Output> yoloV7Detection(String test_img) throws Exception {

        // 1. 初始化模型
        // 全局new一次即可，千万不要每次使用都new。可以使用@Bean，或者在spring项目启动时初始化一次即可
        /*Onnx onnx = new YoloV8(names,model_path,false);*/

        // 2. 读取图像
        // 也可以使用接口收到的base64图像Imgcodecs.imdecode()
        Mat img = readOrDownloadImage(test_img);

        // 3. 执行模型推理
        // 这一步已经结束，可以通过接口返回给前端结果，或者自己循环打印看结果输出
        List<Output> outputs = yoloV7Onnx.run(img.clone());

        // 4. 处理并保存图像
        // 可以调用此方法本地查看图片效果，也可以不调用
        yoloV7Onnx.drawprocess(outputs,img);

        return outputs;

    }

    // 下载远程图片并保存为本地临时文件
    public static File downloadImage(String imageUrl) throws Exception {
        URL url = new URL(imageUrl);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("GET");

        InputStream inputStream = connection.getInputStream();
        File tempFile = File.createTempFile("image", ".png");
        FileOutputStream outputStream = new FileOutputStream(tempFile);

        byte[] buffer = new byte[4096];
        int bytesRead;
        while ((bytesRead = inputStream.read(buffer)) != -1) {
            outputStream.write(buffer, 0, bytesRead);
        }

        outputStream.close();
        inputStream.close();

        log.info("保存网络图片到本地的位置路径：{}", tempFile.getAbsolutePath());

        return tempFile;
    }

    // 读取本地或远程图像
    private Mat readOrDownloadImage(String imagePath) throws Exception {
        Mat img;
        if (imagePath.startsWith("http")) {
            // 如果是远程 URL，先下载图片
            File downloadedImage = downloadImage(imagePath);
            img = Imgcodecs.imread(downloadedImage.getAbsolutePath());
        } else {
            // 本地图片
            img = Imgcodecs.imread(imagePath);
        }

        if (img.empty()) {
            throw new Exception("Failed to load image: " + imagePath);
        }

        return img;
    }

}
