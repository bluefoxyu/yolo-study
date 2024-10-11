package com.bluefoxyu.service.impl;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.bluefoxyu.config.ODConfig;
import com.bluefoxyu.domain.ODResult;
import com.bluefoxyu.lock.UserLockManager;
import com.bluefoxyu.service.CameraDetectionWarnService;
import com.bluefoxyu.utils.ImageUtil;
import com.bluefoxyu.utils.Letterbox;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.Lock;


/**
 * <p>
 * 视频流检测接口
 * </p>
 *
 * @author bluefoxyu
 * @since 2024-10-11
 */
@Slf4j
@Service
public class CameraDetectionWarnServiceImpl implements CameraDetectionWarnService {

    // 锁管理器
    private static final UserLockManager userLockManager = new UserLockManager();

    static Map<String, Integer> current = new ConcurrentHashMap<>();
    static String yolov7_model_file = "model/yolov7-tiny.onnx";

    @Override
    public String detectCameraWarning() throws Exception {
        Lock userLock = userLockManager.getLockForCurrentUser(); // 获取当前用户的锁
        userLock.lock(); // 加锁
        try {
            userLockManager.setDetectionStatusForCurrentUser(true); // 设置当前用户状态为正在检测
            nu.pattern.OpenCV.loadLocally();

            // 加载操作系统的 OpenCV 库
            String OS = System.getProperty("os.name").toLowerCase();
            if (OS.contains("win")) {
                System.load(ClassLoader.getSystemResource("lib/opencv_videoio_ffmpeg470_64.dll").getPath());
            }

            // 复制 YOLOv7 模型文件到临时路径
            String modelPath = copyModelToTempFile(yolov7_model_file);
            String[] labels = {
                    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", //... (省略其他标签)
                    "mouse" // 我们感兴趣的标签
            };

            OrtEnvironment environment = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            OrtSession session = environment.createSession(modelPath, sessionOptions);

            session.getInputInfo().keySet().forEach(x -> {
                try {
                    System.out.println("input name = " + x);
                    System.out.println(session.getInputInfo().get(x).getInfo().toString());
                } catch (OrtException e) {
                    throw new RuntimeException(e);
                }
            });

            ODConfig odConfig = new ODConfig();
            VideoCapture video = new VideoCapture();
            video.open(0);

            if (!video.isOpened()) {
                System.err.println("打开视频流失败！");
                String videoPath = ClassLoader.getSystemResource("video/car3.mp4").getPath();
                video.open(videoPath);
            }

            int minDwDh = Math.min((int) video.get(Videoio.CAP_PROP_FRAME_WIDTH), (int) video.get(Videoio.CAP_PROP_FRAME_HEIGHT));
            int thickness = minDwDh / ODConfig.lineThicknessRatio;
            int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;

            Mat img = new Mat();
            int detect_skip = 4;
            int detect_skip_index = 1;
            float[][] outputData = null;
            Mat image;

            Letterbox letterbox = new Letterbox();
            OnnxTensor tensor;

            while (video.read(img)) {
                if ((detect_skip_index % detect_skip == 0) || outputData == null) {
                    image = img.clone();
                    image = letterbox.letterbox(image);
                    Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2RGB);
                    image.convertTo(image, CvType.CV_32FC1, 1. / 255);
                    float[] whc = new float[3 * 640 * 640];
                    image.get(0, 0, whc);
                    float[] chw = ImageUtil.whc2cwh(whc);

                    detect_skip_index = 1;
                    FloatBuffer inputBuffer = FloatBuffer.wrap(chw);
                    tensor = OnnxTensor.createTensor(environment, inputBuffer, new long[]{1, 3, 640, 640});

                    HashMap<String, OnnxTensor> stringOnnxTensorHashMap = new HashMap<>();
                    stringOnnxTensorHashMap.put(session.getInputInfo().keySet().iterator().next(), tensor);
                    OrtSession.Result output = session.run(stringOnnxTensorHashMap);
                    outputData = (float[][]) output.get(0).getValue();
                } else {
                    detect_skip_index++;
                }

                current.clear();
                for (float[] x : outputData) {
                    ODResult odResult = new ODResult(x);
                    String boxName = labels[odResult.getClsId()];
                    current.put(boxName, current.getOrDefault(boxName, 0) + 1);

                    Point topLeft = new Point((odResult.getX0() - letterbox.getDw()) / letterbox.getRatio(),
                            (odResult.getY0() - letterbox.getDh()) / letterbox.getRatio());
                    Point bottomRight = new Point((odResult.getX1() - letterbox.getDw()) / letterbox.getRatio(),
                            (odResult.getY1() - letterbox.getDh()) / letterbox.getRatio());
                    Scalar color = new Scalar(odConfig.getOtherColor(odResult.getClsId()));

                    Imgproc.rectangle(img, topLeft, bottomRight, color, thickness);
                    Point boxNameLoc = new Point((odResult.getX0() - letterbox.getDw()) / letterbox.getRatio(),
                            (odResult.getY0() - letterbox.getDh()) / letterbox.getRatio() - 3);
                    Imgproc.putText(img, boxName, boxNameLoc, fontFace, 0.7, color, thickness);

                }

                HighGui.imshow("result", img);

                // 检查是否需要停止检测
                if (!userLockManager.getDetectionStatusForCurrentUser()) {
                    log.info("检测已被手动停止");
                    break; // 如果用户请求停止，退出循环
                }

                if (HighGui.waitKey(1) != -1) {
                    break;
                }
            }

            HighGui.destroyAllWindows();
            video.release();
            userLockManager.setDetectionStatusForCurrentUser(false); // 检测完成，设置状态为未检测

            return "摄像头检测成功";
        } finally {
            userLock.unlock(); // 解锁
        }
    }

    @Override
    public String stopDetection() {
        if (!userLockManager.getDetectionStatusForCurrentUser()) {
            return "当前用户没有正在进行的检测";
        }
        userLockManager.setDetectionStatusForCurrentUser(false); // 设置为未检测状态
        return "请求停止检测成功";
    }


    // 复制模型文件到临时文件以便加载
    private String copyModelToTempFile(String modelFileName) throws Exception {
        ClassLoader classLoader = getClass().getClassLoader();
        try (InputStream inputStream = classLoader.getResourceAsStream(modelFileName)) {
            if (inputStream == null) {
                throw new IllegalStateException("模型文件未找到: " + modelFileName);
            }

            Path tempFile = Files.createTempFile("onnx-model", ".onnx");
            Files.copy(inputStream, tempFile, StandardCopyOption.REPLACE_EXISTING);
            return tempFile.toString();
        }
    }
}
