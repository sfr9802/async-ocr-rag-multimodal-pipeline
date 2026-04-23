package com.aipipeline.coreapi.storage.adapter.out.s3;

import com.aipipeline.coreapi.common.AipipelineProperties;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.S3Configuration;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;

import java.net.URI;

/**
 * Exposes {@link S3Client} and {@link S3Presigner} beans when
 * {@code aipipeline.storage.backend=s3}. Uses path-style access for
 * MinIO compatibility.
 */
@Configuration
@ConditionalOnProperty(prefix = "aipipeline.storage", name = "backend", havingValue = "s3")
public class S3StorageConfiguration {

    @Bean
    S3Client s3Client(AipipelineProperties properties) {
        var s3Props = properties.storage().s3();
        var creds = StaticCredentialsProvider.create(
                AwsBasicCredentials.create(s3Props.accessKey(), s3Props.secretKey()));
        return S3Client.builder()
                .endpointOverride(URI.create(s3Props.endpoint()))
                .region(Region.of(s3Props.region()))
                .credentialsProvider(creds)
                .serviceConfiguration(S3Configuration.builder()
                        .pathStyleAccessEnabled(true)
                        .build())
                .forcePathStyle(true)
                .build();
    }

    @Bean
    S3Presigner s3Presigner(AipipelineProperties properties) {
        var s3Props = properties.storage().s3();
        var creds = StaticCredentialsProvider.create(
                AwsBasicCredentials.create(s3Props.accessKey(), s3Props.secretKey()));
        return S3Presigner.builder()
                .endpointOverride(URI.create(s3Props.endpoint()))
                .region(Region.of(s3Props.region()))
                .credentialsProvider(creds)
                .serviceConfiguration(S3Configuration.builder()
                        .pathStyleAccessEnabled(true)
                        .build())
                .build();
    }
}
