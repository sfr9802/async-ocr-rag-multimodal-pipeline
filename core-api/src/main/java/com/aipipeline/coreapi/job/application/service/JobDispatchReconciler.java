package com.aipipeline.coreapi.job.application.service;

import com.aipipeline.coreapi.common.TimeProvider;
import com.aipipeline.coreapi.job.application.port.out.JobDispatchPort;
import com.aipipeline.coreapi.job.application.port.out.JobRepository;
import com.aipipeline.coreapi.job.domain.Job;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.List;

/**
 * Repairs the gap between PostgreSQL as source-of-truth and Redis delivery.
 *
 * Job creation dispatches after commit so workers never see a database ghost.
 * If Redis is unavailable after that commit, the job remains QUEUED without a
 * list item. Likewise, if a worker dies after BRPOP, the job remains RUNNING
 * until its claim lease expires. This reconciler periodically re-enqueues both
 * cases; the atomic claim path remains the guard against duplicate execution.
 */
@Service
public class JobDispatchReconciler {

    private static final Logger log = LoggerFactory.getLogger(JobDispatchReconciler.class);
    private static final int DEFAULT_BATCH_SIZE = 100;

    private final JobRepository jobRepository;
    private final JobDispatchPort dispatchPort;
    private final TimeProvider timeProvider;

    public JobDispatchReconciler(JobRepository jobRepository,
                                 JobDispatchPort dispatchPort,
                                 TimeProvider timeProvider) {
        this.jobRepository = jobRepository;
        this.dispatchPort = dispatchPort;
        this.timeProvider = timeProvider;
    }

    @Scheduled(
            fixedDelayString = "${aipipeline.queue.redis.reconcile-interval-ms:10000}",
            initialDelayString = "${aipipeline.queue.redis.reconcile-initial-delay-ms:10000}")
    public void redispatchLostDeliveries() {
        Instant now = timeProvider.now();
        List<Job> candidates = jobRepository.findRedispatchCandidates(now, DEFAULT_BATCH_SIZE);
        if (candidates.isEmpty()) {
            return;
        }

        int dispatched = 0;
        for (Job job : candidates) {
            try {
                dispatchPort.dispatch(job);
                dispatched++;
            } catch (JobDispatchPort.DispatchException ex) {
                log.warn(
                        "Failed to redispatch job {} status={} after reconciliation scan: {}",
                        job.getId(), job.getStatus(), ex.getMessage());
                log.debug("Redispatch failure detail for job {}", job.getId(), ex);
            }
        }

        log.info(
                "Dispatch reconciliation scanned {} candidate(s), redispatched {}",
                candidates.size(), dispatched);
    }
}
