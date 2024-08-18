package com.bluefoxyu.controller;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.bluefoxyu.config.ODConfig;
import com.bluefoxyu.domain.ODResult;
import com.bluefoxyu.utils.ImageUtil;
import com.bluefoxyu.utils.Letterbox;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.nio.FloatBuffer;
import java.util.HashMap;


/**
 * 摄像头识别，这是yolov7的视频识别例子，v5和v8的根据下面的思路，将其他文件中的代码复制过来即可
 * 视频帧率15最佳，20也可以，不建议30，分辨率640最佳，720也可以。不建议1080，码率不要超过2048，1024最佳
 */

@RestController
@RequestMapping("/camera")
public class CameraDetectionController {

    // 用于停止检测
    private static volatile boolean stopRequested = false;
    private static volatile boolean startRequested = false;

    // 视频帧率15最佳，20也可以，不建议30，分辨率640最佳，720也可以。不建议1080，码率不要超过2048，1024最佳 。可在摄像头自带的管理页面中设备，主码流和子码流

    //java -Djava.awt.headless=false -jar CameraDetectionApplication.jar

    //摄像头检测
    @PostMapping("/detect")
    public String CameraDetection() throws OrtException {
        //System.load(ClassLoader.getSystemResource("lib/opencv_videoio_ffmpeg470_64.dll").getPath());
        nu.pattern.OpenCV.loadLocally();

        //linux和苹果系统需要注释这一行，如果仅打开摄像头预览，这一行没有用，可以删除，如果rtmp或者rtsp等等这一样有用，也可以用pom依赖代替
        String OS = System.getProperty("os.name").toLowerCase();
        if (OS.contains("win")) {
            System.load(ClassLoader.getSystemResource("lib/opencv_videoio_ffmpeg470_64.dll").getPath());
        }
        //yolov7的ONNX模型文件路径
        String model_path = "./CameraDetection/src/main/resources/model/yolov7-tiny.onnx";

        // 用于识别的标签，labels 数组包含了模型可以识别的目标类别
        String[] labels = {
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

        // 加载ONNX模型
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();

        // 使用gpu,需要本机按钻过cuda，并修改pom.xml，不安装也能运行本程序
        // sessionOptions.addCUDA(0);
        // 实际项目中，视频识别必须开启GPU，并且要防止队列堆积

        OrtSession session = environment.createSession(model_path, sessionOptions);
        // 输出基本信息
        session.getInputInfo().keySet().forEach(x -> {
            try {
                System.out.println("input name = " + x);
                System.out.println(session.getInputInfo().get(x).getInfo().toString());
            } catch (OrtException e) {
                throw new RuntimeException(e);
            }
        });

        // 加载标签及颜色
        ODConfig odConfig = new ODConfig();
        VideoCapture video = new VideoCapture();

        // 也可以设置为rtmp或者rtsp视频流：video.open("rtmp://192.168.1.100/live/test"), 海康，大华，乐橙，宇视，录像机等等
        // video.open("rtsp://192.168.1.100/live/test")
        // 也可以静态视频文件：video.open("video/car3.mp4");  flv 等
        // 不持支h265视频编码，如果无法播放或者程序卡住，请修改视频编码格式
        video.open(0);  //获取电脑上第0个摄像头
        //video.open("images/car2.mp4"); //不开启gpu比较卡

        //可以把识别后的视频在通过rtmp转发到其他流媒体服务器，就可以远程预览视频后视频，需要使用ffmpeg将连续图片合成flv 等等，很简单。
        if (!video.isOpened()) {
            System.err.println("打开视频流失败,未检测到监控,请先用vlc软件测试链接是否可以播放！,下面试用默认测试视频进行预览效果！");
            video.open("video/car3.mp4");
        }

        // 在这里先定义下框的粗细、字的大小、字的类型、字的颜色(按比例设置大小粗细比较好一些)
        int minDwDh = Math.min((int)video.get(Videoio.CAP_PROP_FRAME_WIDTH), (int)video.get(Videoio.CAP_PROP_FRAME_HEIGHT));
        int thickness = minDwDh / ODConfig.lineThicknessRatio;
        double fontSize = minDwDh / ODConfig.fontSizeRatio;
        int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;

        Mat img = new Mat();

        // 跳帧检测，一般设置为3，毫秒内视频画面变化是不大的，快了无意义，反而浪费性能
        int detect_skip = 4;

        // 跳帧计数
        int detect_skip_index = 1;

        // 最新一帧也就是上一帧推理结果
        float[][] outputData   = null;

        //当前最新一帧。上一帧也可以暂存一下
        Mat image;

        Letterbox letterbox = new Letterbox();
        OnnxTensor tensor;

        // 使用多线程和GPU可以提升帧率，线上项目必须多线程！！！,一个线程拉流，将图像存到[定长]队列或数组或者集合，一个线程模型推理，中间通过变量或者队列交换数据,代码示例仅仅使用单线程
        while (video.read(img)) {
            if ((detect_skip_index % detect_skip == 0) || outputData == null){
                // 缓存上一帧图像
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

                // 运行推理
                // 模型推理本质是多维矩阵运算，而GPU是专门用于矩阵运算，占用率低，如果使用cpu也可以运行，可能占用率100%属于正常现象，不必纠结。
                OrtSession.Result output = session.run(stringOnnxTensorHashMap);

                // 得到结果,缓存结果
                try{
                    outputData = (float[][]) output.get(0).getValue();
                }catch (OrtException e){}
            }else{
                detect_skip_index = detect_skip_index + 1;
            }
            // 画框和文字
            for(float[] x : outputData){

                ODResult odResult = new ODResult(x);
                // 业务逻辑写在这里，注释下面代码，增加自己的代码，根据返回识别到的目标类型，编写告警逻辑。等等
                // 实际项目中建议不要在视频画面上画框和文字，只告警，或者在告警图片上画框。画框和文字对视频帧率影响非常大
                // 画框
                Point topLeft = new Point((odResult.getX0() - letterbox.getDw()) / letterbox.getRatio(), (odResult.getY0() - letterbox.getDh()) / letterbox.getRatio());
                Point bottomRight = new Point((odResult.getX1() - letterbox.getDw()) / letterbox.getRatio(), (odResult.getY1() - letterbox.getDh()) / letterbox.getRatio());
                Scalar color = new Scalar(odConfig.getOtherColor(odResult.getClsId()));

                Imgproc.rectangle(img, topLeft, bottomRight, color, thickness);
                // 框上写文字
                String boxName = labels[odResult.getClsId()];
                Point boxNameLoc = new Point((odResult.getX0() - letterbox.getDw()) / letterbox.getRatio(), (odResult.getY0() - letterbox.getDh()) / letterbox.getRatio() - 3);

                // 也可以二次往视频画面上叠加其他文字或者数据，比如物联网设备数据等等
                Imgproc.putText(img, boxName, boxNameLoc, fontFace, 0.7, color, thickness);
                // System.out.println(odResult+"   "+ boxName);

            }

            // 保存告警图像到同级目录
            // Imgcodecs.imwrite(ODConfig.savePicPath, img);
            //服务器部署：由于服务器没有桌面，所以无法弹出画面预览，主要注释一下代码
            HighGui.imshow("result", img);

            startRequested=true;

            if (stopRequested) {
                break;
            }

            // 多次按任意按键关闭弹窗画面，结束程序
            // 需要使用键盘的键入，不能使用鼠标手动关闭
            if(HighGui.waitKey(1) != -1){
                break;
            }

        }

        HighGui.destroyAllWindows();
        video.release();
        stopRequested = false;  // 重置状态以便下一次调用
        startRequested=false;   // 重置状态以便下一次调用
        //System.exit(0);

        System.out.println("该次检测成功~~");

        return "摄像头检测成功";
    }

    @PostMapping("/stop")
    public String stopDetection() {
        if (!startRequested){
            return "没有正在进行的检测";
        }
        stopRequested = true;
        System.out.println("执行停止检测");
        return "请求停止检测成功";
    }

}
