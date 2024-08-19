package com.bluefoxyu.domain;

public class CarDetection extends Detection{

    /**是否为单行车牌**/
    public boolean single;

    /**车牌旋转角度**/
    public float angle;

    /**车牌文本信息**/
    public String plateNo;
    /**车牌的颜色信息**/
    public String plateColor;

    public CarDetection(String label, Integer clsId, float[] bbox, float confidence, boolean single, float angle, String plateNo, String plateColor) {
        super(label, clsId, bbox, confidence);
        this.single = single;
        this.angle = angle;
        this.plateNo = plateNo;
        this.plateColor = plateColor;
    }

    public CarDetection(boolean single, float angle, String plateNo, String plateColor) {
        this.single = single;
        this.angle = angle;
        this.plateNo = plateNo;
        this.plateColor = plateColor;
    }

    public boolean isSingle() {
        return single;
    }

    public void setSingle(boolean single) {
        this.single = single;
    }

    public float getAngle() {
        return angle;
    }

    public void setAngle(float angle) {
        this.angle = angle;
    }

    public String getPlateNo() {
        return plateNo;
    }

    public void setPlateNo(String plateNo) {
        this.plateNo = plateNo;
    }

    public String getPlateColor() {
        return plateColor;
    }

    public void setPlateColor(String plateColor) {
        this.plateColor = plateColor;
    }
}
