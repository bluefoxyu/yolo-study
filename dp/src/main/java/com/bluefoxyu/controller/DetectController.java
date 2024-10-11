package com.bluefoxyu.controller;

import com.bluefoxyu.output.Output;
import com.bluefoxyu.service.DetectService;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
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
    static String test_img = "https://bluefox-web-cos-1319400004.cos.ap-guangzhou.myqcloud.com/images/ebc0e178-15c8-444a-bdd0-fae86e3430e5.png";

    @PostMapping("/yoloV8/detect")
    public List<Output> yoloV8Detection(@RequestBody String photoLoad) throws Exception {
        log.info("yoloV8检测开始");
        return detectService.yoloV8Detection(photoLoad);
    }

    @PostMapping("/yoloV7/detect")
    public List<Output> yoloV7Detection(@RequestBody String photoLoad) throws Exception {
        log.info("yoloV7检测开始");
        return detectService.yoloV7Detection(photoLoad);
    }


}
