package com.aipipeline.coreapi.catalog.domain;

public enum IndexBuildStatus {
    CREATED,
    BUILDING,
    EVALUATING,
    EVAL_PASSED,
    EVAL_FAILED,
    PROMOTED,
    ROLLED_BACK,
    FAILED
}
