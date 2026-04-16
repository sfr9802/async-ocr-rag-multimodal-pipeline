package com.aipipeline.coreapi.common;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Typed configuration bound from application.yml under the {@code aipipeline}
 * root. Keeps scattered {@code @Value} lookups out of services.
 */
@ConfigurationProperties(prefix = "aipipeline")
public record AipipelineProperties(
        Storage storage,
        Queue queue,
        Worker worker,
        Claim claim
) {

    public record Storage(
            String backend,
            Local local,
            S3 s3,
            SignedUrl signedUrl
    ) {
        public record Local(String rootDir) {}
        public record S3(
                String endpoint,
                String region,
                String bucket,
                String accessKey,
                String secretKey
        ) {}
        public record SignedUrl(long ttlSeconds) {}
    }

    public record Queue(
            String backend,
            RedisQueue redis
    ) {
        public record RedisQueue(String pendingKey, String inflightKey) {}
    }

    public record Worker(String callbackBaseUrl) {}

    public record Claim(long leaseSeconds) {}
}
