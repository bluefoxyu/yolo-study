package com.bluefoxyu.lock;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;


/**
 * <p>
 * 用户锁，用于视频流检测
 * </p>
 *
 * @author bluefoxyu
 * @since 2024-10-11
 */
public class UserLockManager {
    // 存储用户的锁，使用ConcurrentHashMap以确保线程安全
    private static final ConcurrentHashMap<String, Lock> userLocks = new ConcurrentHashMap<>();

    // 存储用户的检测状态，true表示正在检测，false表示未检测
    private static final ConcurrentHashMap<String, Boolean> userDetectionStatus = new ConcurrentHashMap<>();

    /**
     * 获取当前用户的锁，如果当前用户没有锁则创建一个新的ReentrantLock
     * @return 当前用户的锁
     */
    public Lock getLockForCurrentUser() {
        Long currentUserId = 1L; // 获取当前用户ID
        // 使用computeIfAbsent方法，如果当前用户的锁不存在，则创建一个新的ReentrantLock
        return userLocks.computeIfAbsent(String.valueOf(currentUserId), id -> new ReentrantLock());
    }

    /**
     * 获取当前用户的检测状态
     * @return 当前用户的检测状态，默认为false
     */
    public Boolean getDetectionStatusForCurrentUser() {
        Long currentUserId = 1L; // 获取当前用户ID
        // 返回当前用户的检测状态，如果没有记录则返回false
        return userDetectionStatus.getOrDefault(String.valueOf(currentUserId), false);
    }

    /**
     * 设置当前用户的检测状态
     * @param status 检测状态，true表示正在检测，false表示未检测
     */
    public void setDetectionStatusForCurrentUser(boolean status) {
        Long currentUserId = 1L; // 获取当前用户ID
        // 更新当前用户的检测状态
        userDetectionStatus.put(String.valueOf(currentUserId), status);
    }
}
