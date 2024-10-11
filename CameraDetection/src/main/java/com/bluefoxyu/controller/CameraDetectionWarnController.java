package com.bluefoxyu.controller;

import com.bluefoxyu.service.CameraDetectionWarnService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 摄像头识别，告警判断示例
 */

@Slf4j
@RestController()
@RequestMapping("/web/user/camera-detection-warn")
public class CameraDetectionWarnController {

    @Autowired
    private CameraDetectionWarnService cameraDetectionWarnService;
    @PostMapping("/detect")
    public String CameraDetectionWarn() {
        try {
            return cameraDetectionWarnService.detectCameraWarning();
        } catch (Exception e) {
            log.error("摄像头检测失败: {}", e.getMessage());
            return "摄像头检测失败: " + e.getMessage();
        }
    }

    @PostMapping("/stop")
    public String stopDetection() {
        return cameraDetectionWarnService.stopDetection();
    }
}
