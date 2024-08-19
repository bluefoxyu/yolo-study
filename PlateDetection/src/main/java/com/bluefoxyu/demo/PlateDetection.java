package com.bluefoxyu.demo;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.bluefoxyu.config.ODConfig;
import com.bluefoxyu.domain.CarDetection;
import com.bluefoxyu.utils.ImageUtil;
import com.bluefoxyu.utils.Letterbox;
import org.opencv.core.Point;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 车牌识别，不准确可以自己收集新数据集训练，这里只提供demo，想要高精度，自己训练AI模型
 *
 * 思路：先用检测模型识别画面中是否存在车牌，在什么位置
 * 再用根据检测模型返回的车牌坐标，将车牌裁剪出来，给第二个文字识别模型，第二模型用ocr也可以
 *
 * 标准情况下还需要识别车牌倾斜角度，进行调度矫正
 */
public class PlateDetection {

    static {
        // 加载opencv动态库，
        //System.load(ClassLoader.getSystemResource("lib/opencv_videoio_ffmpeg470_64.dll").getPath());
        nu.pattern.OpenCV.loadLocally();
    }

    final static String[] PLATE_COLOR = new String[]{"黑牌", "蓝牌", "绿牌", "白牌", "黄牌"};
    final static String PLATE_NAME= "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品";

    public static void main(String[] args) throws OrtException {

        // 车牌检测模型
        String model_path1 = "./PlateDetection/src/main/resources/model/plate_detect.onnx";

        // 车牌识别模型
        String model_path2 = "./PlateDetection/src/main/resources/model/plate_rec_color.onnx";

        // 要检测的图片所在目录
        String imagePath = "./yolo-common/src/main/java/com/bluefoxyu/carImg";

        float confThreshold = 0.35F;

        float nmsThreshold = 0.45F;

        // 1.单行蓝牌
        // 2.单行黄牌
        // 3.新能源车牌
        // 4.白色警用车牌
        // 5.教练车牌
        // 6.武警车牌
        // 7.双层黄牌
        // 8.双层白牌
        // 9.使馆车牌
        // 10.港澳粤Z牌
        // 11.双层绿牌
        // 12.民航车牌
        String[] labels = {"1", "2", "3", "4", "5", "6", "7","8", "9", "10", "11","12"};

        // 加载ONNX模型
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        OrtSession session = environment.createSession(model_path1, sessionOptions);

        // 加载ONNX模型
        OrtEnvironment environment2 = OrtEnvironment.getEnvironment();
        OrtSession session2 = environment2.createSession(model_path2, sessionOptions);

        // 加载标签及颜色
        ODConfig odConfig = new ODConfig();
        Map<String, String> map = getImagePathMap(imagePath);
        for(String fileName : map.keySet()){
            String imageFilePath = map.get(fileName);
            System.out.println(imageFilePath);
            // 读取 image
            Mat img = Imgcodecs.imread(imageFilePath);
            Mat image = img.clone();
            Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2RGB);

            // 在这里先定义下框的粗细、字的大小、字的类型、字的颜色(按比例设置大小粗细比较好一些)
            int minDwDh = Math.min(img.width(), img.height());
            int thickness = minDwDh/ODConfig.lineThicknessRatio;
            long start_time = System.currentTimeMillis();
            // 更改 image 尺寸
            Letterbox letterbox = new Letterbox();
            image = letterbox.letterbox(image);

            double ratio  = letterbox.getRatio();
            double dw = letterbox.getDw();
            double dh = letterbox.getDh();
            int rows  = letterbox.getHeight();
            int cols  = letterbox.getWidth();
            int channels = image.channels();

            image.convertTo(image, CvType.CV_32FC1, 1. / 255);
            float[] whc = new float[3 * 640 * 640];
            image.get(0, 0, whc);
            float[] chw = ImageUtil.whc2cwh(whc);

            // 创建OnnxTensor对象
            long[] shape = { 1L, (long)channels, (long)rows, (long)cols };
            OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(chw),shape );
            HashMap<String, OnnxTensor> stringOnnxTensorHashMap = new HashMap<>();
            stringOnnxTensorHashMap.put(session.getInputInfo().keySet().iterator().next(), tensor);

            // 运行推理
            OrtSession.Result output = session.run(stringOnnxTensorHashMap);
            float[][] outputData = ((float[][][])output.get(0).getValue())[0];
            Map<Integer, List<float[]>> class2Bbox = new HashMap<>();
            for (float[] bbox : outputData) {
                float score = bbox[4];
                if (score < confThreshold) continue;

                float[] conditionalProbabilities = Arrays.copyOfRange(bbox, 5, bbox.length);
                int label = argmax(conditionalProbabilities);

                // xywh to (x1, y1, x2, y2)
                xywh2xyxy(bbox);

                // 去除无效结果
                if (bbox[0] >= bbox[2] || bbox[1] >= bbox[3]) continue;

                class2Bbox.putIfAbsent(label, new ArrayList<>());
                class2Bbox.get(label).add(bbox);
            }

            List<CarDetection> CarDetections = new ArrayList<>();
            for (Map.Entry<Integer, List<float[]>> entry : class2Bbox.entrySet()) {

                List<float[]> bboxes = entry.getValue();
                bboxes = nonMaxSuppression(bboxes, nmsThreshold);
                for (float[] bbox : bboxes) {
                    String labelString = labels[entry.getKey()];
                    CarDetections.add(new CarDetection(labelString,entry.getKey(), Arrays.copyOfRange(bbox, 0, 4), bbox[4],bbox[13] == 0,0.0f,null,null));
                }
            }


            for (CarDetection carDetection : CarDetections) {
                float[] bbox = carDetection.getBbox();

                Rect rect = new Rect(new Point((bbox[0]-dw)/ratio, (bbox[1]-dh)/ratio), new Point((bbox[2]-dw)/ratio, (bbox[3]-dh)/ratio));
                // img.submat(rect)
                Mat image2 = new Mat(img.clone(), rect);
                Imgproc.cvtColor(image2, image2, Imgproc.COLOR_BGR2RGB);
                Letterbox letterbox2 = new Letterbox(168,48);
                image2 = letterbox2.letterbox(image2);

                double ratio2  = letterbox2.getRatio();
                double dw2 = letterbox2.getDw();
                double dh2 = letterbox2.getDh();
                int rows2  = letterbox2.getHeight();
                int cols2  = letterbox2.getWidth();
                int channels2 = image2.channels();

                image2.convertTo(image2, CvType.CV_32FC1, 1. / 255);
                float[] whc2 = new float[3 * 168 * 48];
                image2.get(0, 0, whc2);
                float[] chw2 = ImageUtil.whc2cwh(whc2);

                // 创建OnnxTensor对象
                long[] shape2 = { 1L, (long)channels2, (long)rows2, (long)cols2 };
                OnnxTensor tensor2 = OnnxTensor.createTensor(environment2, FloatBuffer.wrap(chw2), shape2);
                HashMap<String, OnnxTensor> stringOnnxTensorHashMap2 = new HashMap<>();
                stringOnnxTensorHashMap2.put(session2.getInputInfo().keySet().iterator().next(), tensor2);

                // 运行推理
                OrtSession.Result output2 = session2.run(stringOnnxTensorHashMap2);
                float[][][] result = (float[][][]) output2.get(0).getValue();
                String plateNo = decodePlate(maxScoreIndex(result[0]));
                System.err.println("车牌号码："+plateNo);
                //车牌颜色识别
                float[][] color = (float[][]) output2.get(1).getValue();
                double[] colorSoftMax = softMax(floatToDouble(color[0]));
                Double[] colorRResult = decodeColor(colorSoftMax);
                carDetection.setPlateNo(plateNo);
                carDetection.setPlateColor( PLATE_COLOR[colorRResult[0].intValue()]);

                Point topLeft = new Point((bbox[0]-dw)/ratio, (bbox[1]-dh)/ratio);
                Point bottomRight = new Point((bbox[2]-dw)/ratio, (bbox[3]-dh)/ratio);
                Imgproc.rectangle(img, topLeft, bottomRight, new Scalar(0,255,0), thickness);
                // 框上写文字
                BufferedImage bufferedImage = matToBufferedImage(img);
                Point boxNameLoc = new Point((bbox[0]-dw)/ratio, (bbox[1]-dh)/ratio-3);
                Graphics2D g2d = bufferedImage.createGraphics();
                g2d.setFont(new Font("微软雅黑", Font.PLAIN, 20));
                g2d.setColor(Color.RED);
                g2d.drawString(PLATE_COLOR[colorRResult[0].intValue()]+"-"+plateNo, (int)((bbox[0]-dw)/ratio), (int)((bbox[1]-dh)/ratio-3)); // 假设的文本位置
                g2d.dispose();
                try {
                    ImageIO.write(bufferedImage, "jpg", new File("temp_output_image.jpg"));
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }

            }
            System.out.printf("time：%d ms.", (System.currentTimeMillis() - start_time));

            System.out.println();


            // 弹窗展示图像
            HighGui.imshow("Display Image", Imgcodecs.imread("temp_output_image.jpg"));
            // 按任意按键关闭弹窗画面，结束程序
            HighGui.waitKey();
        }
        HighGui.destroyAllWindows();
        System.exit(0);

    }


    public static void xywh2xyxy(float[] bbox) {
        float x = bbox[0];
        float y = bbox[1];
        float w = bbox[2];
        float h = bbox[3];

        bbox[0] = x - w * 0.5f;
        bbox[1] = y - h * 0.5f;
        bbox[2] = x + w * 0.5f;
        bbox[3] = y + h * 0.5f;
    }

    public static List<float[]> nonMaxSuppression(List<float[]> bboxes, float iouThreshold) {

        List<float[]> bestBboxes = new ArrayList<>();

        bboxes.sort(Comparator.comparing(a -> a[4]));

        while (!bboxes.isEmpty()) {
            float[] bestBbox = bboxes.remove(bboxes.size() - 1);
            bestBboxes.add(bestBbox);
            bboxes = bboxes.stream().filter(a -> computeIOU(a, bestBbox) < iouThreshold).collect(Collectors.toList());
        }

        return bestBboxes;
    }


    // 单纯为了显示中文演示使用，实际项目中用不到这个
    public static BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] b = new byte[bufferSize];
        mat.get(0, 0, b); // 获取所有像素数据
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
    }

    private static int[] maxScoreIndex(float[][] result){
        int[] indexes = new int[result.length];
        for (int i = 0; i < result.length; i++){
            int index = 0;
            float max = Float.MIN_VALUE;
            for (int j = 0; j < result[i].length; j++) {
                if (max < result[i][j]){
                    max = result[i][j];
                    index = j;
                }
            }
            indexes[i] = index;
        }
        return indexes;
    }

    public static float computeIOU(float[] box1, float[] box2) {

        float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);

        float left = Math.max(box1[0], box2[0]);
        float top = Math.max(box1[1], box2[1]);
        float right = Math.min(box1[2], box2[2]);
        float bottom = Math.min(box1[3], box2[3]);

        float interArea = Math.max(right - left, 0) * Math.max(bottom - top, 0);
        float unionArea = area1 + area2 - interArea;
        return Math.max(interArea / unionArea, 1e-8f);

    }

    private static Double[] decodeColor(double[] indexes){
        double index = -1;
        double max = Double.MIN_VALUE;
        for (int i = 0; i < indexes.length; i++) {
            if (max < indexes[i]){
                max = indexes[i];
                index = i;
            }
        }
        return new Double[]{index, max};
    }



    public static double [] floatToDouble(float[] input){
        if (input == null){
            return null;
        }
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++){
            output[i] = input[i];
        }
        return output;
    }

    private static String decodePlate(int[] indexes){
        int pre = 0;
        StringBuffer sb = new StringBuffer();
        for(int index : indexes){
            if(index != 0 && pre != index){
                sb.append(PLATE_NAME.charAt(index));
            }
            pre = index;
        }
        return sb.toString();
    }

    //返回最大值的索引
    public static int argmax(float[] a) {
        float re = -Float.MAX_VALUE;
        int arg = -1;
        for (int i = 0; i < a.length; i++) {
            if (a[i] >= re) {
                re = a[i];
                arg = i;
            }
        }
        return arg;
    }


    public static double[] softMax(double[] tensor){
        if(Arrays.stream(tensor).max().isPresent()){
            double maxValue = Arrays.stream(tensor).max().getAsDouble();
            double[] value = Arrays.stream(tensor).map(y-> Math.exp(y - maxValue)).toArray();
            double total = Arrays.stream(value).sum();
            return Arrays.stream(value).map(p -> p/total).toArray();
        }else{
            throw new NoSuchElementException("No value present");
        }
    }
    public static Map<String, String> getImagePathMap(String imagePath){
        Map<String, String> map = new TreeMap<>();
        File file = new File(imagePath);
        if(file.isFile()){
            map.put(file.getName(), file.getAbsolutePath());
        }else if(file.isDirectory()){
            for(File tmpFile : Objects.requireNonNull(file.listFiles())){
                map.putAll(getImagePathMap(tmpFile.getPath()));
            }
        }
        return map;
    }
}
