package com.bluefoxyu.service;

import ai.onnxruntime.OrtException;
import com.bluefoxyu.output.Output;

import java.util.List;

public interface DetectService {
    List<Output> yoloV8Detection(String test_img) throws OrtException;
    List<Output> yoloV7Detection(String test_img) throws OrtException;
}
