package com.bluefoxyu;
 
 
import ai.onnxruntime.*;
import com.alibaba.fastjson.JSONObject;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import sun.font.FontDesignMetrics;
 
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.nio.FloatBuffer;
import java.nio.channels.Channel;
import java.util.*;
import java.util.List;
 
/**
*   @desc : video 视频 yolov5实时推理, 单线程（无缓冲队列）
*   @auth : tyf
*   @date : 2023-09-16  14:37:29
*/
public class yolov5_predict_video2 {

    /*环境与模型初始化：
    静态块中加载了ONNX模型、OpenCV库，并从模型中提取了一些必要的配置信息，
    如模型的输入形状（宽、高、通道数）以及类别标签列表。这里使用的是ONNX格式的YOLOv5模型，并且假设模型的权重文件已经保存在指定路径中。*/
 
    // onnxruntime 环境
    public static OrtEnvironment env;
    public static OrtSession session;
 
    // 模型的类别信息,从权重读取
    public static List<String> clazzStr;
 
    // 模型的输入shape,从权重读取
    public static int count;//1 模型每次处理一张图片
    public static int channels;//3 模型通道数
    public static int netHeight;//640 模型高
    public static int netWidth;//640 模型宽
 
    // 检测框筛选阈值,参考 detect.py 中的设置
    public static float confThreshold = 0.65f;
    public static float nmsThreshold = 0.45f;
 
    // 标注颜色
    public static Scalar color = new Scalar(0, 0, 255);
    public static int tickness = 2;
 
    static {
        try {
 
            String weight = new File("").getCanonicalPath() + "\\model\\deeplearning\\yolov5\\yolov5s.onnx";
            System.out.println("weight的目录为 : " + weight);
            env = OrtEnvironment.getEnvironment();
            session = env.createSession(weight, new OrtSession.SessionOptions());
            OnnxModelMetadata metadata = session.getMetadata();
            Map<String, NodeInfo> infoMap = session.getInputInfo();
            TensorInfo nodeInfo = (TensorInfo)infoMap.get("images").getInfo();
            String nameClass = metadata.getCustomMetadata().get("names");
            JSONObject names = JSONObject.parseObject(nameClass.replace("\"","\"\""));
            clazzStr = new ArrayList<>();
            names.entrySet().forEach(n->{
                clazzStr.add(String.valueOf(n.getValue()));
            });
            count = (int)nodeInfo.getShape()[0];//1 模型每次处理一张图片
            channels = (int)nodeInfo.getShape()[1];//3 模型通道数
            netHeight = (int)nodeInfo.getShape()[2];//640 模型高
            netWidth = (int)nodeInfo.getShape()[3];//640 模型宽
//            System.out.println("模型通道数="+channels+",网络输入高度="+netHeight+",网络输入宽度="+netWidth);
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

    /*目标检测类Detection：
    Detection类用于存储和处理目标检测的结果。
    每个实例表示一个检测框，包括其坐标、类别和置信度。
    calculateIoU方法用于计算两个检测框之间的交并比（IoU），这是在进行非极大值抑制（NMS）时需要用到的。*/

    // 目标框
    public static class  Detection{
        float x1;
        float y1;
        float x2;
        float y2;
        int type_max_index;
        float type_max_value;
        String type_max_name;
        public Detection(float[] box){
            // xywh
            float x = box[0];
            float y = box[1];
            float w = box[2];
            float h = box[3];
            // x1y1x2y2
            this.x1 = x - w * 0.5f;
            this.y1 = y - h * 0.5f;
            this.x2 = x + w * 0.5f;
            this.y2 = y + h * 0.5f;
            // 计算概率最大值index,第5位后面开始就是概率
            int max_index = 0;
            float max_value = 0;
            for (int i = 5; i < box.length; i++) {
                if (box[i] > max_value) {
                    max_value = box[i];
                    max_index = i;
                }
            }
            type_max_index = max_index - 5;
            type_max_value = max_value;
            type_max_name = clazzStr.get(type_max_index);
        }

        // 计算两个交并比
        private static double calculateIoU(Detection box1, Detection box2) {
            double x1 = Math.max(box1.x1, box2.x1);
            double y1 = Math.max(box1.y1, box2.y1);
            double x2 = Math.min(box1.x2, box2.x2);
            double y2 = Math.min(box1.y2, box2.y2);
            double intersectionArea = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);
            double box1Area = (box1.x2 - box1.x1 + 1) * (box1.y2 - box1.y1 + 1);
            double box2Area = (box2.x2 - box2.x1 + 1) * (box2.y2 - box2.y1 + 1);
            double unionArea = box1Area + box2Area - intersectionArea;
            return intersectionArea / unionArea;
        }
    }

    /*图像预处理函数resizeWithPadding和hwc2chw：
    resizeWithPadding方法用于将输入图像按比例缩放至YOLOv5模型所需的输入尺寸，并进行适当的填充，使得输入图像的长宽比保持不变。
    hwc2chw方法用于将图像数据从HWC（Height-Width-Channel）格式转换为CHW格式，这是模型输入所需要的格式。*/

    public static Mat resizeWithPadding(Mat src) {
        Mat dst = new Mat();
        int oldW = src.width();
        int oldH = src.height();
        double r = Math.min((double) netWidth / oldW, (double) netHeight / oldH);
        int newUnpadW = (int) Math.round(oldW * r);
        int newUnpadH = (int) Math.round(oldH * r);
        int dw = (Long.valueOf(netWidth).intValue() - newUnpadW) / 2;
        int dh = (Long.valueOf(netHeight).intValue() - newUnpadH) / 2;
        int top = (int) Math.round(dh - 0.1);
        int bottom = (int) Math.round(dh + 0.1);
        int left = (int) Math.round(dw - 0.1);
        int right = (int) Math.round(dw + 0.1);
        Imgproc.resize(src, dst, new Size(newUnpadW, newUnpadH));
        Core.copyMakeBorder(dst, dst, top, bottom, left, right, Core.BORDER_CONSTANT);
        return dst;
    }
 
    public static float[] hwc2chw(float[] src) {
        float[] chw = new float[src.length];
        int j = 0;
        for (int ch = 0; ch < 3; ++ch) {
            for (int i = ch; i < src.length; i += 3) {
                chw[j] = src[i];
                j++;
            }
        }
        return chw;
    }

    /*推理与标注infer：
    infer方法是整个推理流程的核心部分。它接收一帧图像作为输入，首先对其进行预处理，然后通过ONNX Runtime执行推理操作。推理得到的结果是一个包含多个检测框的二维数组。
    检测结果经过置信度筛选后，进行非极大值抑制（NMS）以去除重复的框。
    最后，将保留的检测框绘制到原始图像上，框内还会显示目标类别和置信度。*/

    /**
    *   @desc : 推理并标注一帧
    *   @auth : tyf
    *   @date : 2023-09-16  15:48:09
    */
    public static long infer(Mat frame){
 
        long ts = System.currentTimeMillis();
 
        // 尺寸转换
        Mat input = resizeWithPadding(frame);
        // BGR -> RGB
        Imgproc.cvtColor(input, input, Imgproc.COLOR_BGR2RGB);
        //  归一化 0-255 转 0-1
        input.convertTo(input, CvType.CV_32FC1, 1. / 255);
 
        // 提起像素
        float[] hwc = new float[ channels * netWidth * netWidth];
        input.get(0, 0, hwc);
        float[] chw = hwc2chw(hwc);
 
        // 输入 tenser 并推理
        try {
            OnnxTensor tensor_input = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{count,channels,netWidth,netHeight});
            OrtSession.Result result = session.run(Collections.singletonMap("images", tensor_input));
            OnnxTensor tensor_output = (OnnxTensor)result.get(0);
 
            // 结果后处理 1,25200,117
            float[][] data = ((float[][][])tensor_output.getValue())[0];
 
            List<Detection> box_before_nsm = new ArrayList<>();
            List<Detection> box_after_nsm = new ArrayList<>();
            for(int i=0;i<data.length;i++){
                float[] obj = data[i];
                if(obj[4]>=confThreshold){
                    box_before_nsm.add(new Detection(obj));
                }
            }
 
            box_before_nsm.sort((o1, o2) -> Float.compare(o2.type_max_value,o1.type_max_value));
            while (!box_before_nsm.isEmpty()){
                Detection maxObj = box_before_nsm.get(0);
                box_after_nsm.add(maxObj);
                Iterator<Detection> it = box_before_nsm.iterator();
                while (it.hasNext()) {
                    Detection obj = it.next();
                    // 计算交并比
                    if(Detection.calculateIoU(maxObj,obj)>nmsThreshold){
                        it.remove();
                    }
                }
            }
 
            // 标注
            box_after_nsm.stream().forEach(n->{
 
                float x1 = n.x1;
                float y1 = n.y1;
                float x2 = n.x2;
                float y2 = n.y2;
 
                // 转为原始坐标
                float[] x1y1x2y2 = xy2xy(frame.width(),frame.height(),new float[]{x1,y1,x2,y2});
                x1 = x1y1x2y2[0];
                y1 = x1y1x2y2[1];
                x2 = x1y1x2y2[2];
                y2 = x1y1x2y2[3];
 
                // 类别和概率
                String clazz = n.type_max_name;
                String percent = String.format("%.2f", n.type_max_value*100)+"%";
 
                // 边框
                Imgproc.rectangle(frame, new Point(x1,y1), new Point(x2,y2), color, tickness);
                // 类别
                putText(frame,clazz+" "+percent,(int)x1,(int)y1-13-tickness,13,Color.BLACK,Color.RED);
 
            });
            tensor_input.close();
            tensor_output.close();
            input.release();
        }
        catch (Exception e){
            e.printStackTrace();
            System.exit(0);
        }
 
        long te = System.currentTimeMillis();
        return te-ts;
    }

    /*坐标转换xy2xy：
    由于输入图像经过了缩放和填充，xy2xy方法用于将检测结果的坐标转换回原始图像的坐标系统。*/
 
    // 原始图像 w1*h1
    // 模型图像 w2*h2
    // 待转换的坐标 x1y1x2y2
    public static float[] xy2xy(int w1,int h1,float[] x1y1x2y2){
 
        float gain = Math.min((float) netWidth / w1, (float) netHeight / h1);
        float padW = (netWidth - w1 * gain) * 0.5f;
        float padH = (netHeight - h1 * gain) * 0.5f;
        float xmin = x1y1x2y2[0];
        float ymin = x1y1x2y2[1];
        float xmax = x1y1x2y2[2];
        float ymax = x1y1x2y2[3];
        float xmin_ = Math.max(0, Math.min(w1 - 1, (xmin - padW) / gain));
        float ymin_ = Math.max(0, Math.min(h1 - 1, (ymin - padH) / gain));
        float xmax_ = Math.max(0, Math.min(w1 - 1, (xmax - padW) / gain));
        float ymax_ = Math.max(0, Math.min(h1 - 1, (ymax - padH) / gain));
        return new float[]{xmin_,ymin_,xmax_,ymax_};
    }


    /*绘制中文文本putText：

    putText方法用于在图像上绘制中文文本，它首先通过Java的图形库生成一个包含文字的BufferedImage，
    然后将这个BufferedImage转换为OpenCV的Mat对象，最后将其绘制到视频帧中指定的位置。*/

    // 绘制中文
    public static void putText(Mat src,String text,int x,int y,int charHeight,Color fontColor,Color backgroundColor){
        // 超出区域
        if(x<0||y<0){
            return;
        }
        // 获取字符串绘制的宽度
        Font font = new Font("Dialog", Font.BOLD, charHeight); // 设置字体和字号
        FontDesignMetrics metrics = FontDesignMetrics.getMetrics(font);
        int textWidth = metrics.stringWidth(text);
        // 创建一个java的空白图片并写入汉字
        BufferedImage image = new BufferedImage(textWidth, charHeight, BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D g2d = image.createGraphics();
        g2d.setColor(backgroundColor); // 设置背景色为白色
        g2d.fillRect(0, 0, textWidth, charHeight); // 填充整个图片区域
        g2d.setFont(font); // 设置绘图字体
        g2d.setColor(fontColor); // 设置文本颜色为黑色
        g2d.drawString(text, 0, Double.valueOf(charHeight*0.85).intValue()); // 在图片上写入汉字
        g2d.dispose(); // 释放绘图资源
        // 转为 mat
        byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        Mat mat = Mat.eye(image.getHeight(), image.getWidth(), CvType.CV_8UC3);
        mat.put(0, 0, pixels);
        // 在原始图片 src 的指定位置绘制 mat
        int colStart = x;
        int colEnd = x + mat.width();
        int rowStart = y;
        int rowEnd = y + mat.height();
        // 限制到图片区域内
        if(x>src.width()||y>src.height()){
            System.out.println("超出区域");
            return;
        }
        if(colEnd>src.width()){
            colEnd = src.width();
        }
        if(rowEnd>src.height()){
            rowEnd = src.height();
        }
        // 截取防止超出
        int sub_x = 0;
        int sub_y = 0;
        int sub_w = colEnd - colStart - 1;
        int sub_h =  rowEnd - rowStart - 1;
        if(sub_w<=0||sub_h<=0){
            System.out.println("无可显示距离");
            return;
        }
        // 创建一个矩形区域,从原始图片中截取
        Rect roi = Imgproc.boundingRect(new MatOfPoint(
                new Point(sub_x,sub_y),
                new Point(sub_x,sub_y+sub_h),
                new Point(sub_x+sub_w,sub_y),
                new Point(sub_x+sub_w,sub_y+sub_h)
        ));
        // 提取子图像
        Mat subImage = new Mat(mat, roi);
        subImage.copyTo(src.submat(rowStart,rowEnd,colStart,colEnd));
    }
 
 
    public static void main(String[] args) throws Exception{
 
 
        // 视频、rtsp流等
        String video = new File("").getCanonicalPath() + "\\model\\deeplearning\\yolov5\\1.mp4";
        System.out.println("video 的目录为 : " + video);
 
        // 创建VideoCapture对象并打开视频文件
        VideoCapture cap = new VideoCapture(video);
 
        // 设置想要的fps,每帧最大休眠时长
        int fps = 30;
        int interval = 1000/fps;
 
        // 视频帧宽高
        int width = (int) cap.get(Videoio.CAP_PROP_FRAME_WIDTH);
        int height = (int) cap.get(Videoio.CAP_PROP_FRAME_HEIGHT);
 
        // 用于显示的面板
        JFrame win = new JFrame("Image");
        JPanel panel = new JPanel();
        panel.setPreferredSize(new Dimension(width, height));
        win.getContentPane().add(panel);
        win.setVisible(true);
        win.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        win.setResizable(true);
        win.pack();
 
        // 用于显示的缓存,要修改图像直接修改 pixels 数组即可
        BufferedImage buffer = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        byte[] pixels = ((DataBufferByte) buffer.getRaster().getDataBuffer()).getData();
 
        // 创建一个Mat对象用于存储每一帧
        Mat frame = new Mat(height, width, CvType.CV_8UC3);
        int realFps = 0; // 真实fps
        int frameIndex = 0; // 当前处于第几帧
        double lastTime = 0; // 上次计算真实fps的时间
        int sleepTime = 0;// 渲染前休眠时间
        long lastDraw = 0;// 上次渲染时间
        long inferTime = 0;// 每fps个帧数的推理总耗时
        long inferTimeTotal = 0;// 每fps个帧数的推理总耗时
 
        // 处理每一帧
        while (cap.read(frame)) {
 
            // 在这里执行帧推理和标注,返回推理耗时
            long use = infer(frame);
 
            inferTimeTotal += use;
 
            // mat 写入到 pixels 像素缓存中,这里基本没有耗时
            frame.get(0,0,pixels);
 
            // 每fps个帧数计算一次总耗时,得到每帧耗时(真实fps)和每帧推理耗时
            if(frameIndex%fps==0){
                double thisTime = System.currentTimeMillis();
                // 真实fps
                realFps = (int)(1000/((thisTime - lastTime)/fps));
                // 计算真实推理耗时,并重置总耗时
                inferTime = inferTimeTotal / fps;
                inferTimeTotal = 0;
                // 保存为上一次统计时间
                lastTime = thisTime;
            }
 
            // 计算左上角显示的每帧间隔休眠时长
            sleepTime = 0;
            while(System.currentTimeMillis()-lastDraw<interval){
                try {
                    // 每次休眠1毫秒,直到下一次渲染时间距离上一次渲染时间保持稳定间隔
                    Thread.sleep(1);
                    sleepTime++;
                } catch (InterruptedException e1) {
                    e1.printStackTrace();
                }
            }
            lastDraw = System.currentTimeMillis();
 
            // 实时渲染,这里基本没有耗时,左上角显示fps和休眠时长
            Graphics2D g2 =(Graphics2D)buffer.getGraphics();
            g2.setColor(Color.BLACK);
            g2.drawString("FPS: "+realFps+"   "+"Sleep: "+sleepTime+"ms   "+"Infer: "+inferTime +"ms", 5, 15);
            panel.getGraphics().drawImage(buffer, 0, 0, panel);
 
            frameIndex++;
 
        }
 
 
 
    }
 
 
}