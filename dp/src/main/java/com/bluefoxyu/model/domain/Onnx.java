package com.bluefoxyu.model.domain;

import ai.onnxruntime.*;
import com.bluefoxyu.output.Output;
import com.bluefoxyu.utils.ImageUtil;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * onnx抽象类，没写get 和 set 自己增加
 */
public abstract class Onnx {

    protected OrtEnvironment environment;

    protected OrtSession session;

    protected String[] labels;

    protected double[][] colors;

    boolean gpu = false;

    long[] input_shape = {1, 3, 640, 640};

    int stride = 32;

    public float confThreshold = 0.45F;

    public OnnxJavaType inputType;

    OnnxTensor inputTensor;

    public float nmsThreshold = 0.45F;

    public double ratio;
    public double dw;
    public double dh;

    /**
     * 初始化
     * @param labels 模型分类标签
     * @param model_path 模型路径
     * @param gpu 是否开启gou
     * @throws OrtException
     */
    /*Onnx类通过OrtEnvironment和OrtSession初始化模型，
    并从模型的输入信息中获取张量（Tensor）的类型（如UINT8或FLOAT）。
    同时，为每个分类标签随机生成一个颜色，用于绘制检测框。*/
    public Onnx(String[] labels,String model_path,boolean gpu) throws OrtException {
        nu.pattern.OpenCV.loadLocally();
        this.labels = labels;
        this.gpu = gpu;
        environment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        if(gpu){
            sessionOptions.addCUDA(0);
        }
        session = environment.createSession(model_path,sessionOptions);
        Map<String, NodeInfo> inputMetaMap = session.getInputInfo();
        NodeInfo inputMeta = inputMetaMap.get(session.getInputNames().iterator().next());
        this.inputType = ((TensorInfo) inputMeta.getInfo()).type;
        System.out.println(inputMeta.toString());


        colors = new double[labels.length][3];
        for (int i = 0; i < colors.length; i++) {
            Random random = new Random();
            double[] color = {random.nextDouble()*256, random.nextDouble()*256, random.nextDouble()*256};
            colors[i] = color;
        }
    }

    public  List<Output>  run(Mat img) throws OrtException {
        Map<String, OnnxTensor> inputContainer = this.preprocess(img);
        return this.postprocess(this.session.run(inputContainer),img);
    }

    /**
     * 后处理
     * @param result
     * @return
     * @throws OrtException
     */
    public abstract List<Output> postprocess(OrtSession.Result result, Mat img) throws OrtException;

    /**
     * 画框标注，可以继承后复写此方法
     * @param outputs
     */
    public Mat drawprocess(List<Output> outputs, Mat img){

       for (Output output : outputs) {
           System.err.println( output.toString());
           Point topLeft = new Point(output.getLocation().get(0).get("x"), output.getLocation().get(0).get("y"));
           Point bottomRight = new Point(output.getLocation().get(2).get("x"), output.getLocation().get(2).get("y"));

           Scalar color = new Scalar(colors[output.getClsId()]);

           Imgproc.rectangle(img, topLeft, bottomRight, color, 2);

           Point boxNameLoc = new Point(output.getLocation().get(0).get("x"), output.getLocation().get(0).get("y"));

           // 也可以二次往视频画面上叠加其他文字或者数据，比如物联网设备数据等等
           Imgproc.putText(img, labels[output.getClsId()], boxNameLoc, Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
       }
        /*System.err.println("----------------------推理成功的图像保存在项目的video目录下：output.png，可以打开查看效果！---------------------------------");
        Imgcodecs.imwrite("video/output.png", img);
        return img;*/
        // 设置保存图像的目录路径
        String outputDir = "./dp/video";
        // 检查目录是否存在，如果不存在则创建
        File directory = new File(outputDir);
        if (!directory.exists()) {
            directory.mkdirs(); // 创建目录及其所有必需的父目录
            System.out.println("没有video目录，创建目录成功");
        }
        // 获取当前日期和时间
        String timeStamp = new SimpleDateFormat("yyyy-MM-dd-HH_mm_ss").format(new Date());
        // 设置保存文件的完整路径
        String outputPath = outputDir + "/output-" + timeStamp + ".png";
        System.err.println("----------------------推理成功的图像保存在项目的video目录下：" + outputPath + "，可以打开查看效果！---------------------------------");
        // 保存图像
        Imgcodecs.imwrite(outputPath, img);
        return img;
    };

    /**
     * 默认预处理方法，如果输入shape不一样可以继承后覆盖重写该方法
     * @param img 图像
     * @return
     * @throws OrtException
     */
    public  Map<String, OnnxTensor> preprocess(Mat img) throws OrtException {
        img = this.letterbox(img);
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2RGB);
        Map<String, OnnxTensor> container = new HashMap<>();

        if (this.inputType.equals(OnnxJavaType.UINT8)) {
            byte[] whc = new byte[(int) (input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3])];
            img.get(0, 0, whc);
            byte[] chw = ImageUtil.whc2cwh(whc);
            ByteBuffer inputBuffer = ByteBuffer.wrap(chw);
            inputTensor = OnnxTensor.createTensor(this.environment, inputBuffer, input_shape, this.inputType);
        } else {

            img.convertTo(img, CvType.CV_32FC1, 1. / 255);
            float[] whc = new float[(int) (input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3])];
            img.get(0, 0, whc);
            float[] chw = ImageUtil.whc2cwh(whc);
            FloatBuffer inputBuffer = FloatBuffer.wrap(chw);
            inputTensor = OnnxTensor.createTensor(this.environment, inputBuffer, input_shape);
        }
        container.put(this.session.getInputInfo().keySet().iterator().next(), inputTensor);
        return container;
    }


    /**
     * 图像缩放
     * @param im
     * @return
     */
    public Mat letterbox(Mat im) {

        int[] shape = {im.rows(), im.cols()};

        double r = Math.min((double) input_shape[2] / shape[0],(double) input_shape[3] / shape[1]);

        Size newUnpad = new Size(Math.round(shape[1] * r), Math.round(shape[0] * r));
        double dw = (double)input_shape[2] - newUnpad.width, dh = (double)input_shape[3] - newUnpad.height;

        dw /= 2;
        dh /= 2;

        if (shape[1] != newUnpad.width || shape[0] != newUnpad.height) {
            Imgproc.resize(im, im, newUnpad, 0, 0, Imgproc.INTER_LINEAR);
        }
        int top = (int) Math.round(dh - 0.1), bottom = (int) Math.round(dh + 0.1);
        int left = (int) Math.round(dw - 0.1), right = (int) Math.round(dw + 0.1);

        Core.copyMakeBorder(im, im, top, bottom, left, right, Core.BORDER_CONSTANT, new Scalar(new double[]{114,114,114}));
        this.ratio = r;
        this.dh = dh;
        this.dw = dw;
        return im;
    }
}
