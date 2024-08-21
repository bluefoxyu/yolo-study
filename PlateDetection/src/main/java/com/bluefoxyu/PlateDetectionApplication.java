package com.bluefoxyu;

import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;

@SpringBootApplication
public class PlateDetectionApplication {
    public static void main(String[] args) {
        SpringApplicationBuilder builder = new SpringApplicationBuilder(PlateDetectionApplication.class);
        builder.headless(false).run(args);
        /*SpringApplication.run(PlateDetectionApplication.class, args);*/
    }
}