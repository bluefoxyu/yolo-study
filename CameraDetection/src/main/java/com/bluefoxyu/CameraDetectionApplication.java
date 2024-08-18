package com.bluefoxyu;


import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;

@SpringBootApplication
public class CameraDetectionApplication {
    public static void main(String[] args) {
        SpringApplicationBuilder builder = new SpringApplicationBuilder(CameraDetectionApplication.class);
        builder.headless(false).run(args);
        /*SpringApplication.run(CameraDetectionApplication.class, args);*/
    }
}