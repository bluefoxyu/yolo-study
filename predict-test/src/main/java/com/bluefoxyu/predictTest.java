package com.bluefoxyu;
 
import ai.onnxruntime.*;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.FloatBuffer;
import java.text.DecimalFormat;
import java.util.*;
 
/**
*   @desc : 使用 com.microsoft.onnxruntime 加载 yolov5 onnx 进行推理
*/
public class predictTest {
    public static OrtEnvironment env;
    public static OrtSession session;
    public static JSONObject names;
    public static long count;
    public static long channels;
    public static long netHeight;
    public static long netWidth;
    public static float confThreshold = 0.25f;
    public static float nmsThreshold = 0.45f;
    static {
        String weight = "C:\\Users\\tyf\\Desktop\\yolov5s.onnx";
        try{
            env = OrtEnvironment.getEnvironment();
            session = env.createSession(weight, new OrtSession.SessionOptions());
            OnnxModelMetadata metadata = session.getMetadata();
            Map<String, NodeInfo> infoMap = session.getInputInfo();
            TensorInfo nodeInfo = (TensorInfo)infoMap.get("images").getInfo();
            String nameClass = metadata.getCustomMetadata().get("names");
            System.out.println("-------打印模型信息开始--------");
            System.out.println("getProducerName="+metadata.getProducerName());
            System.out.println("getGraphName="+metadata.getGraphName());
            System.out.println("getDescription="+metadata.getDescription());
            System.out.println("getDomain="+metadata.getDomain());
            System.out.println("getVersion="+metadata.getVersion());
            System.out.println("getCustomMetadata="+metadata.getCustomMetadata());
            System.out.println("getInputInfo="+infoMap);
            System.out.println("nodeInfo="+nodeInfo);
            System.out.println("-------打印模型信息结束--------");
            names = JSONObject.parseObject(nameClass.replace("\"","\"\""));
            System.out.println("类别信息:"+names);
            count = nodeInfo.getShape()[0];//1 模型每次处理一张图片
            channels = nodeInfo.getShape()[1];//3 模型通道数
            netHeight = nodeInfo.getShape()[2];//640 模型高
            netWidth = nodeInfo.getShape()[3];//640 模型宽
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
 
        }
        catch (Exception e){
            e.printStackTrace();
            System.exit(0);
        }
    }
 
    public static Mat readImg(String path){
        Mat img = Imgcodecs.imread(path);
        return img;
    }
 
 
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
 
    public static OnnxTensor transferTensor(Mat dst){
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);
        dst.convertTo(dst, CvType.CV_32FC1, 1. / 255);
        float[] whc = new float[ Long.valueOf(channels).intValue() * Long.valueOf(netWidth).intValue() * Long.valueOf(netHeight).intValue() ];
        dst.get(0, 0, whc);
        float[] chw = whc2cwh(whc);
        OnnxTensor tensor = null;
        try {
            tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{count,channels,netWidth,netHeight});
        }
        catch (Exception e){
            e.printStackTrace();
            System.exit(0);
        }
        return tensor;
    }
 
    public static float[] whc2cwh(float[] src) {
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
 
    public static int getMaxIndex(float[] array) {
        int maxIndex = 0;
        float maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
 
 
    public static float[] xywh2xyxy(float[] bbox) {
        float x = bbox[0];
        float y = bbox[1];
        float w = bbox[2];
        float h = bbox[3];
        float x1 = x - w * 0.5f;
        float y1 = y - h * 0.5f;
        float x2 = x + w * 0.5f;
        float y2 = y + h * 0.5f;
        return new float[]{
                x1 < 0 ? 0 : x1,
                y1 < 0 ? 0 : y1,
                x2 > netWidth ? netWidth:x2,
                y2 > netHeight? netHeight:y2};
    }
 
    public static JSONArray filterRec1(float[][] data){
        JSONArray recList = new JSONArray();
        for (float[] bbox : data){
            float[] xywh = new float[] {bbox[0],bbox[1],bbox[2],bbox[3]};
            float[] xyxy = xywh2xyxy(xywh);
            float confidence = bbox[4];
            float[] classInfo = Arrays.copyOfRange(bbox, 5, 85);
            int maxIndex = getMaxIndex(classInfo);
            float maxValue = classInfo[maxIndex];
            String maxClass = (String)names.get(Integer.valueOf(maxIndex));
            // 首先根据框图置信度粗选
            if(confidence>=confThreshold){
                JSONObject detect = new JSONObject();
                detect.put("name",maxClass);// 类别
                detect.put("percentage",maxValue);// 概率
                detect.put("xmin",xyxy[0]);
                detect.put("ymin",xyxy[1]);
                detect.put("xmax",xyxy[2]);
                detect.put("ymax",xyxy[3]);
                recList.add(detect);
            }
        }
        return recList;
    }
 
    public static JSONArray filterRec2(JSONArray data){
        JSONArray res = new JSONArray();
        data.sort(Comparator.comparing(obj->((JSONObject)obj).getString("percentage")).reversed());
        while (!data.isEmpty()){
            JSONObject max = data.getJSONObject(0);
            res.add(max);
            Iterator<Object> it = data.iterator();
            while (it.hasNext()) {
                JSONObject obj = (JSONObject)it.next();
                double iou = calculateIoU(max, obj);
                if (iou > nmsThreshold) {
                    it.remove();
                }
            }
        }
        return res;
    }
 
    private static double calculateIoU(JSONObject box1, JSONObject box2) {
        double x1 = Math.max(box1.getDouble("xmin"), box2.getDouble("xmin"));
        double y1 = Math.max(box1.getDouble("ymin"), box2.getDouble("ymin"));
        double x2 = Math.min(box1.getDouble("xmax"), box2.getDouble("xmax"));
        double y2 = Math.min(box1.getDouble("ymax"), box2.getDouble("ymax"));
        double intersectionArea = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);
        double box1Area = (box1.getDouble("xmax") - box1.getDouble("xmin") + 1) * (box1.getDouble("ymax") - box1.getDouble("ymin") + 1);
        double box2Area = (box2.getDouble("xmax") - box2.getDouble("xmin") + 1) * (box2.getDouble("ymax") - box2.getDouble("ymin") + 1);
        double unionArea = box1Area + box2Area - intersectionArea;
        return intersectionArea / unionArea;
    }
 
    public static JSONArray transferSrc2Dst(JSONArray data,int srcw,int srch){
        JSONArray res = new JSONArray();
        float gain = Math.min((float) netWidth / srcw, (float) netHeight / srch);
        float padW = (netWidth - srcw * gain) * 0.5f;
        float padH = (netHeight - srch * gain) * 0.5f;
        data.stream().forEach(n->{
            JSONObject obj = JSONObject.parseObject(n.toString());
            float xmin = obj.getFloat("xmin");
            float ymin = obj.getFloat("ymin");
            float xmax = obj.getFloat("xmax");
            float ymax = obj.getFloat("ymax");
            float xmin_ = Math.max(0, Math.min(srcw - 1, (xmin - padW) / gain));
            float ymin_ = Math.max(0, Math.min(srch - 1, (ymin - padH) / gain));
            float xmax_ = Math.max(0, Math.min(srcw - 1, (xmax - padW) / gain));
            float ymax_ = Math.max(0, Math.min(srch - 1, (ymax - padH) / gain));
            obj.put("xmin",xmin_);
            obj.put("ymin",ymin_);
            obj.put("xmax",xmax_);
            obj.put("ymax",ymax_);
            res.add(obj);
        });
        return res;
    }
    public static void pointBox(String pic,JSONArray box){
        if(box.size()==0){
            System.out.println("暂无识别目标");
            return;
        }
        try {        
            File imageFile = new File(pic);
            BufferedImage img = ImageIO.read(imageFile);
            Graphics2D graph = img.createGraphics();
            graph.setStroke(new BasicStroke(2));
            graph.setFont(new Font("Serif", Font.BOLD, 20));
            graph.setColor(Color.RED);
            box.stream().forEach(n->{
                JSONObject obj = JSONObject.parseObject(n.toString());
                String name = obj.getString("name");
                float percentage = obj.getFloat("percentage");
                float xmin = obj.getFloat("xmin");
                float ymin = obj.getFloat("ymin");
                float xmax = obj.getFloat("xmax");
                float ymax = obj.getFloat("ymax");
                float w = xmax - xmin;
                float h = ymax - ymin;
                graph.drawRect(
                        Float.valueOf(xmin).intValue(), 
                        Float.valueOf(ymin).intValue(),
                        Float.valueOf(w).intValue(),
                        Float.valueOf(h).intValue());
                DecimalFormat decimalFormat = new DecimalFormat("#.##");
                String percentString = decimalFormat.format(percentage);
                graph.drawString(name+" "+percentString, xmin-1, ymin-5);
            });
            graph.dispose();
            JFrame frame = new JFrame("Image Dialog");
            frame.setSize(img.getWidth(), img.getHeight());
            JLabel label = new JLabel(new ImageIcon(img));
            frame.getContentPane().add(label);
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        }
        catch (Exception e){
            e.printStackTrace();
            System.exit(0);
        }
    }
    public static void main(String[] args) throws Exception{
        String pic = "C:\\Users\\tyf\\Desktop\\img.png";
        Mat src = readImg(pic);
        int srcw = src.width();
        int srch = src.height();
        Mat dst = resizeWithPadding(src);
        OnnxTensor tensor = transferTensor(dst);
        OrtSession.Result result = session.run(Collections.singletonMap("images", tensor));
        OnnxTensor res = (OnnxTensor)result.get(0);
        float[][][] dataRes = (float[][][])res.getValue();
        float[][] data = dataRes[0];
        JSONArray srcRec = filterRec1(data);
        JSONArray srcRec2 = filterRec2(srcRec);
        JSONArray dstRec = transferSrc2Dst(srcRec2,srcw,srch);
        pointBox(pic,dstRec);
    }
}