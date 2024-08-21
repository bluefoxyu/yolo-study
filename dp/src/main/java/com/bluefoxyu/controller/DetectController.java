package com.bluefoxyu.controller;


import ai.onnxruntime.OrtException;
import com.bluefoxyu.output.Output;
import com.bluefoxyu.service.DetectService;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@Slf4j
@RestController
@RequestMapping("/api")
public class DetectController {

    @Resource
    private DetectService detectService;

    //这里到时候可以按需求从前端传过来
    static String test_img = "./dp/images/some_people.png";

    @PostMapping("/yoloV8/detect")
    public List<Output> yoloV8Detection() throws OrtException {
        log.info("yoloV8检测开始");
        return detectService.yoloV8Detection(test_img);
    }

    @PostMapping("/yoloV7/detect")
    public List<Output> yoloV7Detection() throws OrtException {
        log.info("yoloV7检测开始");
        return detectService.yoloV7Detection(test_img);
    }


}
