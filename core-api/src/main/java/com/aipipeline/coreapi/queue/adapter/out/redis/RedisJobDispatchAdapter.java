package com.aipipeline.coreapi.queue.adapter.out.redis;

import com.aipipeline.coreapi.common.AipipelineProperties;
import com.aipipeline.coreapi.job.application.port.out.JobDispatchPort;
import com.aipipeline.coreapi.job.domain.Job;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

/**
 * Redis list-based dispatch adapter. Uses LPUSH so the worker's BRPOP /
 * BLPOP can consume in FIFO order (worker does BRPOP from the "right"
 * while we push on the "left").
 *
 * Phase 1: single global pending list. Priority lanes and per-capability
 * queues can be added later by sharding the key.
 */
@Component
public class RedisJobDispatchAdapter implements JobDispatchPort {

    private static final Logger log = LoggerFactory.getLogger(RedisJobDispatchAdapter.class);

    private final StringRedisTemplate redis;
    private final ObjectMapper objectMapper;
    private final String pendingKey;
    private final String callbackBaseUrl;

    public RedisJobDispatchAdapter(StringRedisTemplate redis,
                                   ObjectMapper objectMapper,
                                   AipipelineProperties properties) {
        this.redis = redis;
        this.objectMapper = objectMapper;
        this.pendingKey = properties.queue().redis().pendingKey();
        this.callbackBaseUrl = properties.worker().callbackBaseUrl();
    }

    @Override
    public void dispatch(Job job) {
        QueueMessage message = new QueueMessage(
                job.getId().value(),
                job.getCapability().name(),
                job.getAttemptNo(),
                System.currentTimeMillis(),
                callbackBaseUrl);
        String json;
        try {
            json = objectMapper.writeValueAsString(message);
        } catch (JsonProcessingException ex) {
            throw new DispatchException("Failed to serialize queue message", ex);
        }
        try {
            redis.opsForList().leftPush(pendingKey, json);
            log.info("Dispatched job {} onto {}", job.getId(), pendingKey);
        } catch (Exception ex) {
            throw new DispatchException("Failed to push to Redis queue " + pendingKey, ex);
        }
    }
}
