package com.aipipeline.coreapi.job.adapter.out.persistence;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import java.time.Instant;

/**
 * JPA projection of the Job aggregate. Kept separate from the domain Job
 * class so that Hibernate annotations do not leak into the domain layer.
 */
@Entity
@Table(name = "job")
public class JobJpaEntity {

    @Id
    @Column(name = "id", nullable = false, length = 64)
    private String id;

    @Column(name = "capability", nullable = false, length = 64)
    private String capability;

    @Column(name = "status", nullable = false, length = 32)
    private String status;

    @Column(name = "attempt_no", nullable = false)
    private int attemptNo;

    @Column(name = "claim_token", length = 128)
    private String claimToken;

    @Column(name = "claimed_at")
    private Instant claimedAt;

    @Column(name = "claim_expires_at")
    private Instant claimExpiresAt;

    @Column(name = "last_callback_id", length = 128)
    private String lastCallbackId;

    @Column(name = "error_code", length = 64)
    private String errorCode;

    @Column(name = "error_message", length = 2000)
    private String errorMessage;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    @Column(name = "updated_at", nullable = false)
    private Instant updatedAt;

    protected JobJpaEntity() {}

    public JobJpaEntity(String id, String capability, String status, int attemptNo,
                        String claimToken, Instant claimedAt, Instant claimExpiresAt,
                        String lastCallbackId, String errorCode, String errorMessage,
                        Instant createdAt, Instant updatedAt) {
        this.id = id;
        this.capability = capability;
        this.status = status;
        this.attemptNo = attemptNo;
        this.claimToken = claimToken;
        this.claimedAt = claimedAt;
        this.claimExpiresAt = claimExpiresAt;
        this.lastCallbackId = lastCallbackId;
        this.errorCode = errorCode;
        this.errorMessage = errorMessage;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt;
    }

    public String getId() { return id; }
    public String getCapability() { return capability; }
    public String getStatus() { return status; }
    public int getAttemptNo() { return attemptNo; }
    public String getClaimToken() { return claimToken; }
    public Instant getClaimedAt() { return claimedAt; }
    public Instant getClaimExpiresAt() { return claimExpiresAt; }
    public String getLastCallbackId() { return lastCallbackId; }
    public String getErrorCode() { return errorCode; }
    public String getErrorMessage() { return errorMessage; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }

    public void setCapability(String capability) { this.capability = capability; }
    public void setStatus(String status) { this.status = status; }
    public void setAttemptNo(int attemptNo) { this.attemptNo = attemptNo; }
    public void setClaimToken(String claimToken) { this.claimToken = claimToken; }
    public void setClaimedAt(Instant claimedAt) { this.claimedAt = claimedAt; }
    public void setClaimExpiresAt(Instant claimExpiresAt) { this.claimExpiresAt = claimExpiresAt; }
    public void setLastCallbackId(String lastCallbackId) { this.lastCallbackId = lastCallbackId; }
    public void setErrorCode(String errorCode) { this.errorCode = errorCode; }
    public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
    public void setUpdatedAt(Instant updatedAt) { this.updatedAt = updatedAt; }
}
