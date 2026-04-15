package com.aipipeline.coreapi.job.application.port.out;

import com.aipipeline.coreapi.job.domain.Job;

/**
 * Outbound port for dispatching a job onto the work queue.
 *
 * Phase 1 adapter: Redis list (LPUSH). Replaceable in later phases with any
 * backend (Kafka, SQS, RabbitMQ, Cloud Tasks...) without touching the
 * application layer.
 *
 * The dispatch is intentionally fire-and-forget from the application's
 * perspective: state-of-truth stays in PostgreSQL, and Redis is just the
 * delivery channel. Workers are expected to call back through the
 * {@code JobExecutionUseCase} before the job is considered started.
 */
public interface JobDispatchPort {

    /**
     * Enqueue the given job for worker consumption.
     *
     * @throws DispatchException if the queue backend rejects the message
     */
    void dispatch(Job job);

    class DispatchException extends RuntimeException {
        public DispatchException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
