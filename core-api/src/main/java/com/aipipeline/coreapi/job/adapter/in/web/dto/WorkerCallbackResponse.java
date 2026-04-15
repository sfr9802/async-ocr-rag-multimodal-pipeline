package com.aipipeline.coreapi.job.adapter.in.web.dto;

import com.aipipeline.coreapi.job.application.port.in.JobExecutionUseCase;

public record WorkerCallbackResponse(
        boolean applied,
        boolean duplicate,
        String currentStatus
) {
    public static WorkerCallbackResponse from(JobExecutionUseCase.CallbackResult r) {
        return new WorkerCallbackResponse(
                r.applied(),
                r.duplicate(),
                r.currentStatus() == null ? null : r.currentStatus().name());
    }
}
