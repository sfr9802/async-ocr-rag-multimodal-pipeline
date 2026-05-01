package com.aipipeline.coreapi.job.application.service;

import com.aipipeline.coreapi.common.TimeProvider;
import com.aipipeline.coreapi.job.application.port.out.JobDispatchPort;
import com.aipipeline.coreapi.job.application.port.out.JobRepository;
import com.aipipeline.coreapi.job.domain.Job;
import com.aipipeline.coreapi.job.domain.JobCapability;
import org.junit.jupiter.api.Test;

import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.List;

import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class JobDispatchReconcilerTest {

    private static final Instant NOW = Instant.parse("2026-05-02T00:00:00Z");

    @Test
    void redispatches_candidates_returned_by_repository() {
        JobRepository repository = mock(JobRepository.class);
        JobDispatchPort dispatch = mock(JobDispatchPort.class);
        Job job = queuedJob();
        when(repository.findRedispatchCandidates(NOW, 100))
                .thenReturn(List.of(job));

        var reconciler = new JobDispatchReconciler(
                repository,
                dispatch,
                fixedTimeProvider());

        reconciler.redispatchLostDeliveries();

        verify(dispatch).dispatch(job);
    }

    @Test
    void keeps_scanning_when_one_dispatch_fails() {
        JobRepository repository = mock(JobRepository.class);
        JobDispatchPort dispatch = mock(JobDispatchPort.class);
        Job first = queuedJob();
        Job second = queuedJob();
        when(repository.findRedispatchCandidates(NOW, 100))
                .thenReturn(List.of(first, second));
        doThrow(new JobDispatchPort.DispatchException(
                "redis down", new RuntimeException("boom")))
                .when(dispatch).dispatch(first);

        var reconciler = new JobDispatchReconciler(
                repository,
                dispatch,
                fixedTimeProvider());

        reconciler.redispatchLostDeliveries();

        verify(dispatch).dispatch(first);
        verify(dispatch).dispatch(second);
    }

    private static Job queuedJob() {
        Job job = Job.createNew(JobCapability.MOCK, NOW);
        job.markQueued(NOW);
        return job;
    }

    private static TimeProvider fixedTimeProvider() {
        return new TimeProvider(Clock.fixed(NOW, ZoneOffset.UTC));
    }
}
