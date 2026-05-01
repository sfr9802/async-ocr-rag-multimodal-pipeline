package com.aipipeline.coreapi;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * Entrypoint for the AI processing platform core API.
 *
 * Responsibilities:
 *   - Accept user job submissions (text or file upload)
 *   - Own the job state machine (PENDING / QUEUED / RUNNING / SUCCEEDED / FAILED)
 *   - Register input and output artifacts
 *   - Dispatch work to the worker via a queue port (Redis in phase 1)
 *   - Accept worker claim and callback requests
 *   - Serve artifact access URLs to clients
 *
 * The worker is a separate long-lived process; see ai-worker/.
 */
@SpringBootApplication
@ConfigurationPropertiesScan("com.aipipeline.coreapi")
@EnableScheduling
public class CoreApiApplication {

    public static void main(String[] args) {
        SpringApplication.run(CoreApiApplication.class, args);
    }
}
